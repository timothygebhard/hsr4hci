"""
Utility functions for computing match fractions and the selection mask.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Tuple
from typing import Union

from abel.tools.polar import reproject_image_into_polar
from astropy.convolution import convolve
from skimage.feature import blob_log, canny, match_template
from skimage.morphology import binary_dilation, convex_hull_object, disk
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

import h5py
import numpy as np

from hsr4hci.coordinates import cartesian2polar
from hsr4hci.coordinates import get_center
from hsr4hci.forward_modeling import add_fake_planet
from hsr4hci.general import find_closest, rotate_position
from hsr4hci.general import pad_array_to_shape
from hsr4hci.masking import (
    get_positions_from_mask,
    remove_connected_components,
)
from hsr4hci.psf import get_psf_fwhm


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS RELATED TO MATCH FRACTION
# -----------------------------------------------------------------------------


def get_signal_times_from_keys(keys: list) -> np.ndarray:
    return np.array(
        sorted(list(map(int, filter(lambda _: _.isdigit(), keys))))
    )


def get_all_match_fractions(
    dict_or_path: Union[dict, Union[str, Path]],
    roi_mask: np.ndarray,
    hypotheses: np.ndarray,
    parang: np.ndarray,
    psf_template: np.ndarray,
    frame_size: Tuple[int, int],
    n_roi_splits: int = 1,
    roi_split: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Initialize array for the match fractions (mean and median)
    mean_mfs = np.full(frame_size, np.nan)
    median_mfs = np.full(frame_size, np.nan)

    # Define an array in which we keep track of the "affected pixels" (i.e.,
    # the planet traces for every hypothesis), mostly for debugging purposes
    affected_pixels = np.full(frame_size + frame_size, np.nan)

    # Define positions for which to run (= subset of the ROI)
    positions = get_positions_from_mask(roi_mask)[roi_split::n_roi_splits]

    # We can either work with a dict that holds all results...
    if isinstance(dict_or_path, dict):

        # Get signal times
        signal_times = get_signal_times_from_keys(list(dict_or_path.keys()))

        # Loop over (subset of) ROI and compute match fractions
        for position in tqdm(positions, ncols=80):
            mean_mf, median_mf, affected_mask = get_match_fraction(
                position=position,
                hypothesis=hypotheses[position[0], position[1]],
                results=dict_or_path,
                parang=parang,
                psf_template=psf_template,
                signal_times=signal_times,
                frame_size=frame_size,
            )
            mean_mfs[position] = mean_mf
            median_mfs[position] = median_mf
            affected_pixels[position] = affected_mask

    # ...or with the path to an HDF file that holds all results.
    elif isinstance(dict_or_path, (str, Path)):

        # Open the HDF file
        with h5py.File(dict_or_path, 'r') as results:

            # Get signal times
            signal_times = get_signal_times_from_keys(list(results.keys()))

            # Loop over (subset of) ROI and compute match fractions
            for position in tqdm(positions, ncols=80):
                mean_mf, median_mf, affected_mask = get_match_fraction(
                    position=position,
                    hypothesis=hypotheses[position[0], position[1]],
                    results=results,
                    parang=parang,
                    psf_template=psf_template,
                    signal_times=signal_times,
                    frame_size=frame_size,
                )
                mean_mfs[position] = mean_mf
                median_mfs[position] = median_mf
                affected_pixels[position] = affected_mask

    else:
        raise ValueError(
            f'dict_or_path must be dict or Path, is {type(dict_or_path)}!'
        )

    return mean_mfs, median_mfs, affected_pixels


def get_match_fraction(
    position: Tuple[int, int],
    hypothesis: float,
    results: Union[dict, h5py.File],
    parang: np.ndarray,
    psf_template: np.ndarray,
    signal_times: np.ndarray,
    frame_size: Tuple[int, int],
) -> Tuple[float, float, np.ndarray]:
    """
    Compute the match fraction for a single given position.
    """

    n_frames = len(parang)

    # If we do not have a hypothesis for the current position, we can
    # directly return the match fraction as 0
    if np.isnan(hypothesis):
        return 0, 0, np.full(frame_size, False)

    # Compute the expect final position based on the hypothesis that the
    # signal is at `position` at time `signal_time`
    final_position = rotate_position(
        position=position[::-1],  # position is in numpy coordinates
        center=get_center(frame_size),
        angle=float(parang[int(hypothesis)]),
    )

    # Compute the *full* expected signal stack under this hypothesis and
    # normalize it (so that the thresholding below works more reliably!)
    expected_stack = add_fake_planet(
        stack=np.zeros((n_frames,) + frame_size),
        parang=parang,
        psf_template=psf_template,
        polar_position=cartesian2polar(
            position=(final_position[0], final_position[1]),
            frame_size=frame_size,
        ),
        magnitude=0,
        extra_scaling=1,
        dit_stack=1,
        dit_psf_template=1,
        return_planet_positions=False,
        interpolation='bilinear',
    )
    expected_stack = np.asarray(expected_stack / np.max(expected_stack))

    # Find mask of all pixels that are affected by the planet trace, i.e.,
    # all pixels that at some point in time contain planet signal.
    # The threshold value of 0.2 is a bit of a magic number: it serves to
    # pick only those pixels affected by the central peak of the signal,
    # and not the secondary maxima, which are often too low to be picked
    # up by the HSR, and including them would lower the match fraction.
    affected_mask = np.max(expected_stack, axis=0).astype(float) > 0.2

    # Keep track of the matches (similarity scores) for affected positions
    matches = []

    # Convert signal_times to list (to avoid mypy issue with find_closest())
    signal_times_list = list(signal_times)

    # Loop over all affected positions and check how well the residuals
    # match the expected signals
    for (x, y) in get_positions_from_mask(affected_mask):

        # Skip the hypothesis itself: We know that is is a match, because
        # otherwise it would not be our hypothesis
        if (x, y) == position:
            continue

        # Find the time at which this pixel is affected the most
        peak_time = int(np.argmax(expected_stack[:, x, y]))
        _, peak_time = find_closest(signal_times_list, peak_time)

        # Define shortcuts for the time series that we compare
        a = expected_stack[:, x, y]
        b = np.array(results[str(peak_time)]['residuals'][:, x, y])

        # In case we do not have a signal masking residual for the
        # current affected position, we skip it
        if np.isnan(b).any():
            continue

        # Compute the similarity between the expected signal and the
        # actual "best" residual (and make sure it is between 0 and 1)
        similarity = float(
            np.clip(
                cosine_similarity(X=a.reshape(1, -1), Y=b.reshape(1, -1)),
                a_min=0,
                a_max=1,
            )
        )
        matches.append(similarity)

    # Compute mean and median match fraction for current position
    if matches:
        match_fraction__mean = np.nanmean(matches)
        match_fraction__median = np.nanmedian(matches)
    else:
        match_fraction__mean = 0
        match_fraction__median = 0

    return match_fraction__mean, match_fraction__median, affected_mask


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS RELATED TO SELECTION MASK
# -----------------------------------------------------------------------------

def get_radial_gradient_mask(
    mask_size: Tuple[int, int], power: float = 1.0
) -> np.ndarray:
    """
    Compute radial gradient, that is, a array where the value is its
    separation from the center (to the power of `power`). This can be
    used to re-weight the match fraction to take into account that the
    "uncertainty" for pixels far from the center is smaller than for
    pixels close to the star.

    Args:
        mask_size: A tuple `(x_size, y_size)` specifying the size of
            the radial gradient mask.
        power: The power to which the gradient is taken (default: 1).

    Returns:
        A radial gradient mask of the given size.
    """

    sx, sy = mask_size
    x, y = np.ogrid[0:sx, 0:sy]
    r = np.hypot(x - sx / 2 + 0.5, y - sy / 2 + 0.5)

    return np.asarray(r ** power)


def get_selection_mask(
    match_fraction: np.ndarray,
    parang: np.ndarray,
    psf_template: np.ndarray,
    resolution: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Based on the `match_fraction`, determine the `selection_mask`, that
    is, the mask that decides for which pixels we use the default model
    and for which we use the model based in signal fitting / masking.

    Ideally, it would be sufficient to simply threshold the match
    fraction to obtain the selection mask. In practice, however, this
    does not always work well.
    Therefore, this function uses the following, more complex heuristic:

        1. Convert the match fraction from Cartesian to polar
           coordinates (r, theta).
        2. In polar coordinates, the planet signal is translation
           invariant; and even more importantly, we know exactly how
           it should look like. We can, therefore, compute a cross-
           correlation with the expected signal.
        3. In the result of this template matching, we can find peaks,
           which correspond to the (r, theta) position of the planet
           signals in the match fraction.
        4. Based on these peaks, we explicitly construct the selection
           mask. These masks are more interpretable and will not contain
           any "random" pixels or shapes, unlike the masks that can be
           obtained using the simple thresholding approach.

    To further improve the result, some additional tricks (e.g., re-
    weighting of match fraction using a radial gradient) are used which
    have proven useful in preliminary experiments.

    Args:
        match_fraction: 2D numpy array containing the match fraction.
        parang: 1D numpy array containing the parallactic angles.
        psf_template: 2D numpy array containing the PSF template.
        resolution: Additional factor that is used when upscaling the
            resolution when projecting from Cartesian to polar
            coordinates. Increasing this value can sometimes improve
            the results, but will slow down the computation.

    Returns:
        A 2D numpy array containing a binary mask that can be used to
        select the pixels for which the residuals based on signal
        fitting / signal masking should be used.
    """

    # -------------------------------------------------------------------------
    # Preparations
    # -------------------------------------------------------------------------

    # Compute field rotation; check that it is physically meaningful
    field_rotation = abs(parang[-1] - parang[0])
    if field_rotation > 180:
        raise RuntimeError('field_rotation is greater than 180 degrees!')

    # Prepare the Cartesian match fraction; define shortcuts
    cartesian = np.copy(match_fraction)
    cartesian = np.nan_to_num(cartesian)
    x_size, y_size = cartesian.shape
    center = get_center(cartesian.shape)

    # Multiply the match fraction with a gradient mask that aims to (partially)
    # compensates for the fact that at small separations, only few pixels
    # contribute to the match fraction, meaning it is "easier" to get a
    # high match fraction at small separations, whereas for pixels at larger
    # separation, this "uncertainty" is smaller.
    cartesian *= get_radial_gradient_mask(cartesian.shape)

    # Project the match fraction from Cartesian to polar coordinates.
    # This has the big advantage that the expected signature from a signal
    # will become translation-invariant (i.e., look the same everywhere); in
    # particular, it will take on a shape that we *know* and can search for!
    polar, r_grid, theta_grid = reproject_image_into_polar(
        data=cartesian,
        dr=1.0 / resolution,
        dt=0.1 / resolution,
    )
    r_size, theta_size = theta_grid.shape
    polar /= np.max(polar)

    # -------------------------------------------------------------------------
    # Prepare the template of the expected signal for the cross-correlation
    # -------------------------------------------------------------------------

    # Compute parameters that determine the (approximate) expected size of
    # the signal in the polar map
    psf_fwhm = get_psf_fwhm(psf_template)
    thickness = int(psf_fwhm * r_size / x_size)
    width = int((field_rotation / 360) * theta_size)

    # Compute the (approximate) expected signal
    expected_signal = np.ones((thickness, width - int(thickness / 2)))
    expected_signal = pad_array_to_shape(expected_signal, polar.shape)
    expected_signal = convolve(expected_signal, psf_template.T)
    expected_signal /= np.max(expected_signal)

    # -------------------------------------------------------------------------
    # Cross-correlate polar match fraction with expected signal template
    # -------------------------------------------------------------------------

    # Cross-correlate the polar match fraction with the expected signal
    matched_1 = match_template(
        image=polar,
        template=expected_signal,
        pad_input=True,
        mode='wrap',
    )
    # noinspection PyTypeChecker
    matched_1 = np.clip(matched_1, a_min=0, a_max=None)

    # Run an edge detector and a convex hull filter on the polar match
    # fraction. Ideally, this only leaves (approximately) rectangular blobs
    # at the locations of the planet traces. Finally, remove blobs that are
    # clearly smaller than expected for a planet signal.
    processed_polar = canny(polar, sigma=resolution)
    processed_polar = convex_hull_object(processed_polar)
    processed_polar = remove_connected_components(
        mask=processed_polar, minimum_size=int(thickness * width * 0.75)
    )

    # Cross-correlate the processed polar match fraction
    matched_2 = match_template(
        image=processed_polar,
        template=expected_signal,
        pad_input=True,
        mode='wrap',
    )
    # noinspection PyTypeChecker
    matched_2 = np.clip(matched_2, a_min=0, a_max=None)

    # Compute the average between the matched filter result with and without
    # the edge detector + convex hull trick. Preliminary studies have shown
    # that this can help to suppress large-scale, spatial "background" in the
    # match fraction (because this gets eliminated in the edge detector)
    matched = 0.5 * (matched_1 + matched_2)

    # -------------------------------------------------------------------------
    # Find blobs in the output of the matched filter; create selection_mask
    # -------------------------------------------------------------------------

    # Prepare an empty selection mask
    selection_mask = np.zeros((x_size, y_size))

    # Loop over two offset for the global phase: The choice of the phase for
    # the polar representation is arbitrary, and if we are unlucky, we might
    # have a planet signal that is "split into two", that, half the signal is
    # on the left-hand edge of the polar representation, and the other half is
    # on the right-hand edge.
    # For the template matching, this is not an issue, because of mode='wrap'.
    # However, if a blob gets split into two parts, the blob-finding algorithm
    # might now recognize it properly. Therefore, we run the blob finder twice
    # for different global phase offsets (i.e., we shift the output of the
    # matched filter by 180 degrees) and combine the results.
    for global_offset in (0, np.pi):

        # Shift the matched filter result by 0 or pi
        matched_offset = np.roll(
            a=matched,
            shift=int(matched.shape[1] / 2 * global_offset / np.pi),
            axis=1,
        )

        # Find blobs / peaks in the matched filter result
        # Depending on the data set and / or the hyper-parameters of the HSR,
        # the parameters of the blob finder might require additional tuning.
        blobs = blob_log(
            image=matched_offset,
            max_sigma=5 * resolution,
            num_sigma=20,
            threshold=0.05,
            overlap=0,
            exclude_border=5 * resolution,
        )

        # Loop over blobs and and create and 1-pixel-wide arc for each
        for blob in blobs:

            # Unpack blob coordinates and adjust for the resolution and
            # coordinate system conventions
            rho, phi, _ = blob
            rho /= resolution
            phi = 2 * np.pi * (phi / matched.shape[1] + 0.25)

            # Create a 1-pixel-wide arc centered on the position of the blob,
            # with an opening angle that matches the field rotation
            alpha = np.deg2rad(field_rotation / 2)
            for offset in np.linspace(-alpha, alpha, 100):
                x = int(rho * np.cos(phi + offset + global_offset) + center[0])
                y = int(rho * np.sin(phi + offset + global_offset) + center[1])
                selection_mask[y, x] = 1

    # Dilate the 1 pixel arcs to the expected width of a planet trace to
    # create the final selection mask
    selection_mask = binary_dilation(
        image=selection_mask, selem=disk(int(psf_fwhm / 2 + 0.5))
    )

    return selection_mask, polar, matched, expected_signal
