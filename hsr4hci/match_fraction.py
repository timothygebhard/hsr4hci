"""
Methods for computing match fractions.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Dict, Tuple

from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

import numpy as np

from hsr4hci.coordinates import get_center, cartesian2polar
from hsr4hci.forward_modeling import add_fake_planet
from hsr4hci.general import find_closest, rotate_position
from hsr4hci.masking import get_positions_from_mask


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_all_match_fractions(
    residuals: Dict[str, np.ndarray],
    roi_mask: np.ndarray,
    hypotheses: np.ndarray,
    parang: np.ndarray,
    psf_template: np.ndarray,
    frame_size: Tuple[int, int],
    n_roi_splits: int = 1,
    roi_split: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This is essentially a convenience function which wraps the loop over
    the ROI and calls :func:`get_match_fraction_for_position()` for
    every spatial pixel.

    Args:
        residuals: A dictionary containing the full residuals as they
            are produced by :func:`hsr4hci.training.train_all_models`.
        hypotheses: A 2D numpy array containing the hypotheses map.
        parang: A 1D numpy array of shape `(n_frames, )` containing the
            parallactic angle for every frame.
        psf_template: A 2D numpy array containing the unsaturated PSF
            template.
        frame_size: A tuple `(x_size, y_size)` containing the spatial
            size of the input stack in pixels.
        n_roi_splits: Total number of splits for the ROI if we want to
            compute the match fraction map in parallel.
        roi_split: Index of the ROI split that we want to process here.

    Returns:
        A 3-tuple consisting of

        1. ``mean_mfs``: A 2D numpy array containing the match fraction
           map when using the mean to average.
        2. ``median_mfs``: A 2D numpy array containing the match
           fraction map when using the median to average.
        3. ``affected_pixels``: A 4D numpy array containing which, for
           each position `(x, y)` contains a 2D binary mask with the
           affected mask (see :func:`get_match_fraction_for_position`).
    """

    # Initialize array for the match fractions (mean and median)
    mean_mfs = np.full(frame_size, np.nan)
    median_mfs = np.full(frame_size, np.nan)

    # Define an array in which we keep track of the "affected pixels" (i.e.,
    # the planet traces for every hypothesis), mostly for debugging purposes
    affected_pixels = np.full(frame_size + frame_size, np.nan)

    # Define positions for which to run (= subset of the ROI)
    positions = get_positions_from_mask(roi_mask)[roi_split::n_roi_splits]

    # Get signal times based on the keys of the given results dictionary
    _digit_keys = filter(lambda _: _.isdigit(), residuals.keys())
    signal_times = np.array(sorted(list(map(int, _digit_keys))))

    # Loop over (subset of) ROI and compute match fractions
    for position in tqdm(positions, ncols=80):
        mean_mf, median_mf, affected_mask = get_match_fraction_for_position(
            position=position,
            hypothesis=hypotheses[position[0], position[1]],
            residuals=residuals,
            parang=parang,
            psf_template=psf_template,
            signal_times=signal_times,
            frame_size=frame_size,
        )
        mean_mfs[position] = mean_mf
        median_mfs[position] = median_mf
        affected_pixels[position] = affected_mask

    return mean_mfs, median_mfs, affected_pixels


def get_match_fraction_for_position(
    position: Tuple[int, int],
    hypothesis: float,
    residuals: Dict[str, np.ndarray],
    parang: np.ndarray,
    psf_template: np.ndarray,
    signal_times: np.ndarray,
    frame_size: Tuple[int, int],
) -> Tuple[float, float, np.ndarray]:
    """
    Compute the match fraction for a single given position.

    Args:
        position: A tuple `(x, y)` specifying the position for which to
            compute the match fraction.
        hypothesis: The hypothesis (= temporal index) for the given
            ``position``. In general, this should be an integer, but
            the type here has to be a ``float`` because the value may
            also be `NaN` (in case there is no hypothesis).
        residuals: A dictionary containing the full residuals as they
            are produced by :func:`hsr4hci.training.train_all_models`.
        parang: A 1D numpy array of shape `(n_frames, )` containing the
            parallactic angle for every frame.
        psf_template: A 2D numpy array containing the unsaturated PSF
            template.
        signal_times: A 1D numpy array of shape `(n_signal_times, )`
            containing the temporal grid.
        frame_size: A tuple `(x_size, y_size)` containing the spatial
            size of the input stack in pixels.

    Returns:
        A 3-tuple consisting of

        1. ``match_fraction__mean``: The match fraction for the given
           target ``position`` when using the mean to average.
        2. ``match_fraction__median``: The match fraction for the given
           target ``position`` when using the median to average.
        3. ``affected_mask``: A 2D numpy array containing a binary mask
           that indicates the pixels from which the match fraction was
           computed (i.e., the pixels that are affected by the planet
           according to the ``hypothesis``).
    """

    # Define shortcut for number of frames
    n_frames = len(parang)

    # If we do not have a hypothesis for the current position, we can
    # directly return the match fraction as 0
    if np.isnan(hypothesis):
        return np.nan, np.nan, np.full(frame_size, False)

    # Compute the expect final position based on the hypothesis that the
    # signal is at `position` at time `signal_time`
    final_position = rotate_position(
        position=position[::-1],  # position is in numpy coordinates
        center=get_center(frame_size),
        angle=float(parang[int(hypothesis)]),
    )

    # Compute the *full* expected signal stack under this hypothesis and
    # normalize it (so that the thresholding below works reliably!)
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
    # The threshold value of 0.5 is a bit of a magic number: it serves to pick
    # only those pixels really affected by the central peak of the signal, and
    # not the secondary maxima. The secondary maxima are often too low to be
    # picked up by the HSR, and including them would lower the match fraction.
    affected_mask = np.max(expected_stack, axis=0).astype(float) >= 0.5

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

        # Find the time at which this pixel is affected the most, and find the
        # closest matching signal time for which we have trained a model and
        # therefore have a residual to compare with
        tmp_peak_time = int(np.argmax(expected_stack[:, x, y]))
        _, peak_time = find_closest(signal_times_list, tmp_peak_time)

        # Define shortcuts for the time series that we compare
        a = expected_stack[:, x, y]
        b = np.asarray(residuals[str(peak_time)][:, x, y])

        # In case we do not have a (signal masking) residual for the
        # current affected position, we skip it
        if np.isnan(b).any():
            continue

        # Compute the cosine similarity between the expected signal and the
        # "best" residual as a measure for how well the current pixel (x, y)
        # matches our hypothesis for `position`.
        similarity = cosine_similarity(X=a.reshape(1, -1), Y=b.reshape(1, -1))
        matches.append(float(similarity))

    # Compute mean and median match fraction for current position
    if matches:
        match_fraction__mean = np.nanmean(matches)
        match_fraction__median = np.nanmedian(matches)
    else:
        match_fraction__mean = np.nan
        match_fraction__median = np.nan

    return match_fraction__mean, match_fraction__median, affected_mask
