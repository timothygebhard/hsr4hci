"""
Utility functions that are related to dealing with residuals.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from itertools import product
from math import fmod
from typing import Dict, Tuple, Union

from photutils.centroids import centroid_com
from polarTransform import convertToPolarImage
from skimage.feature import blob_log, match_template

import numpy as np

from hsr4hci.coordinates import get_center
from hsr4hci.general import shift_image, crop_or_pad


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def assemble_residuals_from_hypotheses(
    hypotheses: np.ndarray,
    results: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]],
) -> np.ndarray:
    """
    Assemble the signal masking results based on the `hypotheses`, that
    is, for each spatial pixel use the residual that was obtained when
    assuming the signal time that equals the hypothesis for this pixel.

    Args:
        hypotheses: A 2D numpy array of shape `(width, height)`. Each
            position contains an integers which represents the signal
            time which appears to be the best hypothesis for this pixel
            (i.e., "if there ever is a planet in this pixel, it should
            be at this time").
        results: The dictionary which contains the full results from
            training both the "default" and the signal masking based
            models.

    Returns:
        A 3D numpy array (whose shape matches the stack) containing the
        "best" signal masking-based residuals given the `hypotheses`.
    """

    # Get stack shape from default residuals
    n_frames, x_size, y_size = results['residuals']['default'].shape

    # Initialize the signal masking residuals as all-NaN
    signal_masking_residuals = np.full((n_frames, x_size, y_size), np.nan)

    # Loop over all spatial positions and pick the signal masking-based
    # residual based on the the respective hypothesis for the pixel
    for x, y in product(range(x_size), range(y_size)):
        signal_time = hypotheses[x, y]
        if not np.isnan(signal_time):
            signal_masking_residuals[:, x, y] = np.array(
                results['residuals'][str(int(signal_time))][:, x, y]
            )

    return signal_masking_residuals


def _get_radial_gradient_mask(
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


def _get_expected_signal(
    frame_size: Tuple[int, int],
    field_rotation: float,
    psf_template: np.ndarray,
    grid_size: int,
    relative_rho: float = 0.500,
    relative_phi: float = 0.275,
) -> np.ndarray:

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


def get_residual_selection_mask(
    match_fraction: np.ndarray,
    parang: np.ndarray,
    psf_template: np.ndarray,
    grid_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        grid_size: The size (width and height) of the match fraction
            when projecting to polar coordinates. This value should
            usually be chosen to be larger than the size of the MF in
            Cartesian coordinates (i.e, the usual frame size). Larger
            values can give better results, but slow things down.

    Returns:
        A 5-tuple of numpy array, containing:
        (1) The selection mask, that is, a 2D numpy array containing a
            binary mask that can be used to select the pixels for which
            the residuals based on signal fitting / signal masking
            should be used.
        (2) The polar projection, that is, a 2D numpy array of shape
            `(grid_size, grid_size)` containing the polar projection
            of the match fraction (mostly for debugging purposes).
        (3) The output of the template matching ("heatmaps"), which has
            the same shape as (3). This is the cross-correlation of
            the polar projection with the expected signal.
        (4) The expected signal, that is, a 2D numpy array of shape
            `(grid_size, grid_size)` that contains the (approximate)
            expected planet signal that we would hope to find in the
            polar projections.
        (5) The positions of the (centers) of the planet trace arcs,
            as found by the cross-correlation procedure, that is, a 2D
            numpy array of shape `(N, 2)` where `N` is the number of
            found planet traces.
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
    frame_size = (x_size, y_size)
    center = get_center(cartesian.shape)

    # Multiply the match fraction with a gradient mask that aims to (partially)
    # compensates for the fact that at small separations, only few pixels
    # contribute to the match fraction, meaning it is "easier" to get a
    # high match fraction at small separations, whereas for pixels at larger
    # separation, this "uncertainty" is smaller.
    cartesian *= _get_radial_gradient_mask(cartesian.shape)

    # -------------------------------------------------------------------------
    # Prepare the template of the expected signal for the cross-correlation
    # -------------------------------------------------------------------------

    # Compute the signal that we expect to see in polar coordinates.
    # Note: This signal is not perfectly translation invariant, so the signal
    # that we will be search for is approximately the median of the possible
    # signals that we would expect.
    expected_signal = _get_expected_signal(
        frame_size=frame_size,
        field_rotation=field_rotation,
        psf_template=psf_template,
        grid_size=grid_size,
    )

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
        initialAngle=0,
        finalAngle=2 * np.pi,
    )
    polar = polar.T
    polar /= np.max(polar)

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
    matched = np.clip(matched, a_min=0, a_max=None)

    # -------------------------------------------------------------------------
    # Map to polar coordinates and cross-correlate with the expected signal
    # -------------------------------------------------------------------------

    # Initialize results variables
    selection_mask = np.zeros(frame_size)
    positions = []

    # Loop over two offset for the global phase: The choice of the phase for
    # the polar representation is arbitrary, and if we are unlucky, we might
    # have a planet signal that is "split into two", that, half the signal is
    # on the left-hand edge of the polar representation, and the other half is
    # on the right-hand edge. Therefore, we run with two different phases.
    for global_phase_offset in (0, np.pi)[::-1]:

        # ---------------------------------------------------------------------
        # Run blob finder and construct signal mask
        # ---------------------------------------------------------------------

        # Apply the global phase shift to the outputs of the template matching
        ratio = global_phase_offset / (2 * np.pi)
        phase_shifted_matched = np.roll(
            matched, shift=int(grid_size * ratio), axis=1
        )

        # Find blobs / peaks in the matched filter result
        # Depending on the data set and / or the hyper-parameters of the HSR,
        # the parameters of the blob finder might require additional tuning.
        blobs = blob_log(
            image=phase_shifted_matched,
            max_sigma=grid_size / 4,
            num_sigma=20,
            threshold=0.1,
            overlap=0,
        )

        # Resize the PSF template
        psf_resized = crop_or_pad(psf_template, frame_size)
        psf_resized /= np.max(psf_resized)
        psf_resized = np.asarray(np.clip(psf_resized, 0.2, None)) - 0.2
        psf_resized /= np.max(psf_resized)

        # Initialize the selection mask
        tmp_selection_mask = np.zeros(frame_size)

        # Loop over blobs and and create and 1-pixel-wide arc for each
        for blob in blobs:

            # Unpack blob coordinates
            rho, phi, _ = blob

            # Ignore blobs that are too close to the border
            if phi < grid_size / 4 or phi > 3 * grid_size / 4:
                continue

            # Adjust for coordinate system conventions / scaling
            rho *= min(center[0], center[1]) / grid_size
            phi = (2 * np.pi * phi / grid_size) + global_phase_offset
            phi = fmod(phi, 2 * np.pi)

            # Store the positions
            positions.append((rho, np.rad2deg(phi)))

            # Create an arc at the position defined by (rho, phi): place many
            # shifted copies of the PSF template at the positions of the arc
            alpha = np.deg2rad(field_rotation / 2)
            for offset in np.linspace(-alpha, alpha, 2 * int(field_rotation)):
                x = rho * np.cos(phi + offset)
                y = rho * np.sin(phi + offset)
                shifted = shift_image(psf_resized, (x, y))
                tmp_selection_mask += shifted

        selection_mask += tmp_selection_mask

    # Finally, threshold the selection mask (which so far is a sum of shifted
    # PSF templates) to binarize it
    selection_mask = selection_mask > 0.2

    return selection_mask, polar, matched, expected_signal, np.array(positions)
