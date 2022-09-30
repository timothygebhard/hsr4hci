"""
Methods for dealing with residuals.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from itertools import product
from math import fmod
from typing import Dict, List, Tuple, Union

from astropy.modeling import models, fitting
from photutils.centroids import centroid_com
from polarTransform import convertToPolarImage
from skimage.feature import blob_log, match_template
from skimage.filters import gaussian

import h5py
import numpy as np

from hsr4hci.coordinates import get_center
from hsr4hci.data import get_field_rotation
from hsr4hci.general import shift_image, crop_or_pad
from hsr4hci.masking import mask_frame_around_position


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def assemble_residual_stack_from_hypotheses(
    hypotheses: np.ndarray,
    selection_mask: np.ndarray,
    residuals: Union[h5py.File, Dict[str, np.ndarray]],
) -> np.ndarray:
    """
    Assemble the residual stack based on the ``selection_mask`` and the
    given ``hypotheses``:
    For each spatial pixel where the ``selection_mask`` is `True`, use
    the residual from the model given by the respective entry in
    ``hypotheses``; for all other pixels, use the "default" residual.

    Args:
        hypotheses: A 2D numpy array of shape `(x_size, y_size)`. Each
            position contains an integers which represents the signal
            time which appears to be the best hypothesis for this pixel
            (i.e., "if there ever is a planet in this pixel, it should
            be at this time").
        selection_mask: A 2D numpy array of shape `(x_size, y_size)`
            which contains a mask that determines for which pixels the
            default residual is used and for which pixel the residual
            based on signal fitting / masking is used.
        residuals: The dictionary, or an open HDF file, which contains
            the full results from training both the "default" (vanilla)
            and the models based on signal fitting / masking.

    Returns:
        A 3D numpy array (whose shape matches the stack) containing the
        "best" residuals based on the given the ``hypotheses``.
    """

    # Initialize the result as the default residuals
    result = np.array(residuals['default'])
    _, x_size, y_size = result.shape

    # Loop over all spatial positions and pick the signal masking-based
    # residual based on the respective hypothesis for the pixel
    for x, y in product(np.arange(x_size), np.arange(y_size)):

        # If we do not have a hypothesis, or the selection mask did not select
        # this pixel, do nothing (i.e., keep the "default" residual)
        if np.isnan(hypotheses[x, y]) or (not selection_mask[x, y]):
            continue

        # Otherwise (i.e., if the selection mask is True), replace the default
        # residual with the one that matches the hypothesis for this pixel
        signal_time = str(int(hypotheses[x, y]))
        result[:, x, y] = np.array(residuals[signal_time][:, x, y])

    return result


def _get_expected_signal(
    frame_size: Tuple[int, int],
    field_rotation: float,
    psf_template: np.ndarray,
    grid_size: int,
    relative_rho: float = 0.500,
    relative_phi: float = 0.275,
) -> np.ndarray:
    """
    Auxiliary function to compute the expected signal (or: expected
    signature of a planet) in the polar match fraction map.
    """

    # Define shortcuts
    center = get_center(frame_size)
    rho = min(center[0], center[1]) * relative_rho
    phi = np.pi * (1 + relative_phi)

    # Resize the PSF template
    psf_resized = crop_or_pad(psf_template, frame_size)
    psf_resized /= np.max(psf_resized)
    psf_resized = np.asarray(np.clip(psf_resized, 0.2, None)) - 0.2
    psf_resized /= np.max(psf_resized)

    # Create the expected signal in Cartesian coordinates
    expected_signal_cartesian = np.zeros(frame_size)
    alpha = np.deg2rad(field_rotation / 2)
    for offset in np.linspace(-alpha, alpha, 2 * int(field_rotation)):
        x = rho * np.cos(phi + offset)
        y = rho * np.sin(phi + offset)
        shifted = shift_image(psf_resized, (x, y))
        expected_signal_cartesian += shifted

    # Convert the expected signal to polar coordinates
    expected_signal_polar, _ = convertToPolarImage(
        image=np.nan_to_num(expected_signal_cartesian),
        radiusSize=grid_size,
        angleSize=grid_size,
        initialRadius=0,
        finalRadius=min(center[0], center[1]),
    )
    expected_signal_polar = expected_signal_polar.T
    expected_signal_polar /= np.max(expected_signal_polar)

    # Make sure that the expected signal is properly centered
    polar_center = get_center(expected_signal_polar.shape)
    com = centroid_com(expected_signal_polar)
    d_rho = polar_center[0] - com[0]
    d_phi = polar_center[1] - com[1]
    expected_signal_polar = shift_image(expected_signal_polar, (d_rho, d_phi))

    return np.asarray(expected_signal_polar)


def _prune_blobs(blobs: List[Tuple[float, float, float]]) -> np.ndarray:
    """
    Prune a list of blobs: if there are two blobs at (approximately) the
    same separation, we only keep the brighter blob.

    Note: This is an extremely simple version of "pruning", and it does
    *not* cover all corner cases. However, it seems sufficient for most
    cases that will cause problems in practice.

    Args:
        blobs: A list of 3-tuples describing the blobs we want to prune,
            where each tuple has the form `(separation, polar angle,
            brightness)`.

    Returns:
        A 2D numpy array containing only the positions (separation and
        polar angle) of the pruned list of blobs.
    """

    # Store the pruned blobs that we will return
    pruned = []

    # Loop over all blobs to check which of them we will keep
    for i, blob_1 in enumerate(blobs):

        # Unpack the reference blob; this is the blob for which we
        # decide if we want to keep it or not
        rho_1, phi_1, brightness_1 = blob_1

        # Loop (again) over all blobs
        for j, blob_2 in enumerate(blobs):

            # We do not compare a blob with itself
            if i == j:
                continue

            # Unpack the blob with which we will compare
            rho_2, _, brightness_2 = blob_2

            # If the blob that we are comparing with is radially close and
            # brighter than the reference blob, we break the inner loop
            if abs(rho_1 - rho_2) <= 8 and brightness_2 > brightness_1:
                break

        # Only if the inner for-loop terminated normally, that is, not via the
        # break command, do we add the reference blob
        else:
            pruned.append((rho_1, phi_1))

    return np.array(pruned)


def _refit_blob(
    frame: np.ndarray,
    position: Tuple[float, float],
) -> Tuple[float, float, float]:
    """
    Auxiliary function to fit a blob at the given `position` to refine
    its position and get its brightness / amplitude for pruning.
    """

    # Define the grid for the fit
    x = np.arange(frame.shape[0])
    y = np.arange(frame.shape[1])
    x, y = np.meshgrid(x, y)

    # Create a new Gaussian2D object
    gaussian_model = models.Gaussian2D(x_mean=position[0], y_mean=position[1])

    # Define "search area" by setting minimum and maximum values for the mean
    gaussian_model.x_mean.min = position[0] - 2
    gaussian_model.x_mean.max = position[0] + 2
    gaussian_model.y_mean.min = position[1] - 2
    gaussian_model.y_mean.max = position[1] + 2

    # Mask the frame (set everything to zero that is too far from position)
    masked_frame = mask_frame_around_position(
        frame=np.nan_to_num(frame),
        position=position,
        radius=8,
    )

    # Fit the model to the data
    fit_p = fitting.LevMarLSQFitter()
    gaussian_model = fit_p(gaussian_model, x, y, masked_frame)

    # Get the final position and amplitude of the Gaussian after the fit
    rho = float(gaussian_model.x_mean.value)
    phi = float(gaussian_model.y_mean.value)
    amplitude = float(gaussian_model.amplitude.value)

    return rho, phi, amplitude


def get_gradient_mask(
    grid_size: int,
    frame_size: Tuple[int, int],
    zero_radius: int = 4,
) -> np.ndarray:
    """
    Compute a gradient mask to re-weight the match fraction radially.

    Rationale: The number of "affected pixels" that we use to compute
    a match fraction scales linearly with the separation from the star.
    For pixels very close to the center, only very few pixels contribute
    while for pixels at large separations, the MF is computed as the
    average of many pixels. To reduce both the number of false positives
    at small separations and false negative at large separations, it is
    useful to apply a "gradient mask" to the match fraction to re-weigh
    the MF values based on the separation from the center.

    Args:
        grid_size: Frame size of the polar representation.
        frame_size: Frame size of the Cartesian representation.
        zero_radius: Radius (in pixels) around the center where the
            mask is set to zero; typically 1 FWHM. Even if there is a
            planet this close to the star, we should not be able to
            detect it.

    Returns:
        A gradient mask that can be used to re-weigh the polar MF map.
    """

    # Re-scale the radius between the Cartesian and the polar coordinates
    radius = int(2 * zero_radius / min(frame_size) * grid_size)

    # Construct the gradient mask
    gradient = np.concatenate(
        (np.full(radius, 1e-8), np.linspace(1e-8, 1, grid_size - radius))
    )
    gradient = np.tile(gradient.reshape(-1, 1), reps=(1, grid_size))

    return np.asarray(gradient)


def get_residual_selection_mask(
    match_fraction: np.ndarray,
    parang: np.ndarray,
    psf_template: np.ndarray,
    grid_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Based on the ``match_fraction``, determine the ``selection_mask``,
    that is, the mask that decides for which pixels we use the default
    model and for which we use the model based on signal fitting or
    signal masking.

    Ideally, it would be sufficient to simply threshold the match
    fraction to obtain the selection mask. In practice, however, this
    does not always work well.
    Therefore, this function uses the following, more complex heuristic:

        1. Convert the match fraction from Cartesian to polar
           coordinates $(\rho, \phi)$.
        2. In polar coordinates, the planet signal is translation
           invariant; and even more importantly, we know exactly how
           it should look like. We can, therefore, compute a cross-
           correlation with the expected signal.
        3. In the result of this template matching, we can find peaks,
           which correspond to the $(\rho, \phi)$ position of the planet
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
        grid_size: The size (width and height) of the match fraction
            when projecting to polar coordinates. This value should
            usually be chosen to be larger than the size of the MF in
            Cartesian coordinates (i.e, the usual frame size). Larger
            values can give better results, but slow things down.

    Returns:
        A 5-tuple of numpy array, consisting of

        1. The final ``selection_mask``, that is, a 2D numpy array
           containing a binary mask that can be used to select the
           pixels for which the residuals based on signal fitting or
           signal masking should be used.
        2. The ``polar`` projection, that is, a 2D numpy array of shape
           `(grid_size, grid_size)` containing the polar projection
           of the match fraction (mostly for debugging purposes).
        3. The output of the template matching ("heatmap"), which has
           the same shape as (2). This is the cross-correlation of
           the polar projection with the expected signal.
        4. The template / expected signal, that is, a 2D numpy array of
           shape `(grid_size, grid_size)` that contains the approximate
           expected planet signal that we would hope to find in the
           polar projections.
        5. The positions of the (centers) of the planet trace arcs,
           as found by the cross-correlation procedure, that is, a 2D
           numpy array of shape `(N, 2)` where `N` is the number of
           found planet traces. Each tuple consists of the separation
           and the polar angle (in radian).
    """

    # -------------------------------------------------------------------------
    # Preparations
    # -------------------------------------------------------------------------

    # Prepare the Cartesian match fraction: Make sure all entries are valid
    # numbers with values between 0 and 1.
    cartesian = np.copy(match_fraction)
    cartesian = np.nan_to_num(cartesian)
    cartesian = np.clip(cartesian, a_min=0, a_max=1)

    # Define shortcuts
    x_size, y_size = cartesian.shape
    frame_size = (x_size, y_size)
    center = get_center((cartesian.shape[0], cartesian.shape[1]))

    # Compute field rotation
    field_rotation = get_field_rotation(parang=parang)

    # -------------------------------------------------------------------------
    # Prepare the template of the expected signal for the cross-correlation
    # -------------------------------------------------------------------------

    # Compute the signal that we expect to see in polar coordinates
    expected_signal = _get_expected_signal(
        frame_size=frame_size,
        field_rotation=field_rotation,
        psf_template=psf_template,
        grid_size=grid_size,
    )

    # Add a small negative "glow" around the expected signal. This is needed
    # because otherwise, the template matching (which is really just a cross-
    # correlation) does not pay enough attention to the signal *shape*.
    # Note: The exact parameters here may require some additional fine-tuning.
    negative = np.clip(gaussian(expected_signal, sigma=5), 0, 0.05)
    expected_signal -= negative

    # -------------------------------------------------------------------------
    # Project to polar coordinates
    # -------------------------------------------------------------------------

    # Project the match fraction from Cartesian to polar coordinates
    polar, _ = convertToPolarImage(
        image=np.nan_to_num(cartesian),
        radiusSize=grid_size,
        angleSize=grid_size,
        initialRadius=0,
        finalRadius=min(center[0], center[1]),
        initialAngle=-np.pi,
        finalAngle=np.pi,
    )
    polar = polar.T

    # Compute and apply gradient mask to re-weight the match fraction based on
    # the separation: The number of "affected pixels" that are used to compute
    # the match fraction scales linearly with the separation meaning that high
    # values at large separations are less likely to be just a fluke.
    gradient_mask = get_gradient_mask(
        grid_size=grid_size, frame_size=frame_size, zero_radius=4
    )
    polar *= gradient_mask

    # -------------------------------------------------------------------------
    # Cross-correlate with expected signal
    # -------------------------------------------------------------------------

    # Cross-correlate the polar match fraction with the expected signal
    matched = match_template(
        image=polar,
        template=expected_signal,
        pad_input=True,
        mode='wrap',
    )

    # noinspection PyTypeChecker
    matched = np.array(matched)

    # -------------------------------------------------------------------------
    # Run blob finder, and prune blobs
    # -------------------------------------------------------------------------

    # Loop over two offset for the global phase: The choice of the phase for
    # the polar representation is arbitrary, and if we are unlucky, we might
    # have a planet signal that is "split into two", that, half the signal is
    # on the left-hand edge of the polar representation, and the other half is
    # on the right-hand edge. Therefore, we run with two different phases.
    blobs: List[Tuple[float, float, float]] = []
    for global_phase_offset in (0, np.pi):

        # ---------------------------------------------------------------------
        # Run blob finder and construct signal mask
        # ---------------------------------------------------------------------

        # Apply the global phase shift to the outputs of the template matching
        # and pad it with zeros. The reason for the latter is that the blob
        # finder below sometimes gives wrong results if a blob is too close to
        # the border of the image, and padding with zeros seems like a simple
        # solution for this problem.
        ratio = global_phase_offset / (2 * np.pi)
        phase_shifted_matched = np.roll(
            matched, shift=int(grid_size * ratio), axis=1
        )
        padded_phase_shifted_matched = crop_or_pad(
            phase_shifted_matched, (3 * grid_size, 3 * grid_size)
        )

        # Apply the blob finding algorithm to the matched filter result
        tmp_blobs = blob_log(
            image=padded_phase_shifted_matched,
            min_sigma=0.1,
            max_sigma=grid_size / 4,
            overlap=0.0,
        )

        # Slightly post-process the blobs that we have found:
        # (1) Drop all blobs that are too close to the image border (they will
        #     be found by the other `global_phase_offset` value).
        # (2) Replace the last entry (by default the radius of the blob) by the
        #     brightness of the blob. (Needed for pruning blobs.)
        # (3) Convert the first two entries (the coordinates of the blob in the
        #     polar match fraction map) to values for the radius and separation
        #     in the original (Cartesian) match fraction map.
        for blob in tmp_blobs:

            # Unpack blob coordinates
            rho, phi, _ = blob

            # Undo the zero-padding that we needed for the blob finder
            rho -= grid_size
            phi -= grid_size

            # Ignore blobs that are too close to the border (these will be
            # caught by the respective other `global_phase_offset`)
            if phi < grid_size / 4 or phi > 3 * grid_size / 4:
                continue

            # Re-fit the exact blob position after un-doing the gradient mask
            # and measure the brightness / amplitude for pruning
            phi, rho, brightness = _refit_blob(
                frame=phase_shifted_matched / gradient_mask,
                position=(phi, rho),
            )

            # Adjust for coordinate system conventions / scaling
            rho *= min(center[0], center[1]) / grid_size
            phi = (2 * np.pi * phi / grid_size) + global_phase_offset + np.pi
            phi = fmod(phi, 2 * np.pi)

            # Store the final position and brightness of the blob
            blobs.append((rho, phi, brightness))

    # Prune the list of blobs to get the positions: If there are two planet
    # candidates at the same separation, we should only keep the brighter one.
    # This step is, in principle, optional. However, it helps to reduce false
    # positives because we *know* that it is unlikely to observe two planets
    # at the same separation.
    positions = _prune_blobs(blobs)

    # -------------------------------------------------------------------------
    # Finally, create the selection mask
    # -------------------------------------------------------------------------

    # Resize and normalize the PSF template
    psf_resized = crop_or_pad(psf_template, frame_size)
    psf_resized /= np.max(psf_resized)
    psf_resized = np.asarray(np.clip(psf_resized, 0.2, None)) - 0.2
    psf_resized /= np.max(psf_resized)

    # Initialize the selection mask
    selection_mask = np.zeros(frame_size)

    # Loop over blobs and create an arc at the position defined by (rho, phi):
    # place many shifted copies of the PSF template at the positions of the arc
    for rho, phi in positions:
        alpha = np.deg2rad(field_rotation / 2)
        for offset in np.linspace(-alpha, alpha, 2 * int(field_rotation)):
            x = rho * np.cos(phi + offset)
            y = rho * np.sin(phi + offset)
            shifted = shift_image(psf_resized, (x, y))
            selection_mask += shifted

    # Finally, multiply the selection mask with the (normalized) original match
    # fraction to drop any "bad pixels", and threshold to binarize it to a mask
    selection_mask *= np.nan_to_num(match_fraction) / np.nanmax(match_fraction)
    selection_mask = np.asarray(selection_mask > 0.1)

    return selection_mask, polar, matched, expected_signal, positions
