"""
Utility functions related to positions (injections and references).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import List, Optional, Tuple

from astropy.units import Quantity

import numpy as np


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_injection_position(
    separation: Quantity,
    azimuthal_position: str,
) -> Tuple[Quantity, Quantity]:
    """
    Get the position (in astronomical polar coordinates) at which to
    inject a fake planet; specified by the separation and an identifier
    for the `azimuthal_position`.

    While this function is not particularly complicated, it helps to
    ensure consistency across different parts of the code base.

    Args:
        separation: The separation from the origin / star at the center.
        azimuthal_position: A string specifying the azimuthal position.
            There are six valid positions, denoted by "a", "b", ...,
            "f", which are at positions 9 o'clock, 11 o'clock, ...,
            4 o'clock.
            There are only six such positions, because at the smallest
            (meaningful) separation, exactly 6 apertures can be placed.

    Returns:
        A tuple `(separation, position_angle)` with the desired
        injection position.
    """

    # Convert the identifier to the azimuthal_position ("a", "b", ..., "f")
    # into an angle in (astronomical) polar coordinates; i.e., 0 degree is
    # "up", not "right" (unlike in mathematical polar coordinates).
    lookup_table = {'a': 270, 'b': 330, 'c': 30, 'd': 90, 'e': 150, 'f': 210}
    try:
        position_angle = Quantity(lookup_table[azimuthal_position], 'degree')
    except KeyError:
        raise ValueError('azimuthal_position must be "a", "b", ..., "f"!')

    return separation, position_angle


def get_reference_positions(
    polar_position: Tuple[Quantity, Quantity],
    aperture_radius: Quantity,
    exclusion_angle: Optional[Quantity] = None,
) -> List[Tuple[Quantity, Quantity]]:
    """
    Get a list of reference positions for the given `polar_position`.

    If the `polar_position` is the position of a suspected planet
    signal, then the reference positions can be used to estimate the
    noise at this separation to compute the signal-to-noise ratio.

    Args:
        polar_position: A tuple `(separation, position_angle)` that
            specifies the target position for which the reference
            positions are computed.
        aperture_radius: The radius of the apertures that are used to
            to the photometry. Traditionally, a common choice is to use
            apertures with a diameter of 1 lambda / D. To ensure that
            the samples are actually independent, one should probably
            based the size of the apertures on the size of the PSF of
            the data set (e.g., diameter 1 FWHM).
        exclusion_angle: The angle around the `polar_positions` where
            no reference positions are placed. This can be used, for
            example, to ignore the "neighbors" of the target position
            which, in the case of PCA-based PSF subtraction, often
            contain negative "wings" that would bias the computation of
            the SNR.
            The exclusion angle is centered on the `polar_position`,
            that is, `exclusion_angle / 2` to the left and to the right
            of the `polar_position` do not contain reference positions.
            If `exclusion_angle` is set to None, the exclusion angle is
            automatically determined to exclude the immediate neighbors
            of the `polar_position` (this is the default behavior).

    Returns:
        A list of polar positions (i.e., tuples `(separation,
        position_angle)`) that can be used as reference positions
        for the given `polar_position`.
    """

    # Unpack the polar position
    separation, position_angle = polar_position

    # Compute the "opening angle" that one aperture corresponds to at
    # the given separation
    opening_angle = 2 * np.arcsin(
        aperture_radius / separation.to(aperture_radius.unit)
    )

    # In case no exclusion angle is explicitly specified, compute a default
    # exclusion angle, which is chosen such that the apertures that are the
    # immediate neighbors of the `polar_position` are excluded.
    if exclusion_angle is None:
        exclusion_angle = 3 * opening_angle

    # Ensure that the exclusion angle is always at least as large as the
    # opening angle so that the reference apertures do not overlap with the
    # aperture that is placed at the target `polar_position`
    if exclusion_angle < opening_angle:
        exclusion_angle = opening_angle

    # Compute the "start angle", that is, the angle at which the first
    # reference position is placed
    start_angle = position_angle + exclusion_angle / 2 + opening_angle / 2

    # Compute the "reference angle", that is, the angle of the arc on which
    # we can place the reference positions
    reference_angle = Quantity(360, 'degree') - exclusion_angle - opening_angle

    # Compute the number of reference positions
    # Note: The additional round() call is needed here because otherwise, the
    # result can actually be different on different machines: Depending on the
    # machine `reference_angle / opening_angle` can evaluate to, for example,
    # 4.000000000000001 or 3.9999999999999996 --- and calling int() will then
    # either give 4 or 3.
    n_reference_positions = (
        int(round(float(reference_angle / opening_angle), 1)) + 1
    )

    # Compute the (polar) position angles of the reference positions
    offsets = np.linspace(
        Quantity(0, 'degree'),
        reference_angle.to('degree'),
        n_reference_positions,
    )
    reference_angles = start_angle + offsets

    # Assemble list of reference positions: each reference position is a tuple
    # of the original separation and the reference angle we have just computed
    reference_positions = [(separation, _) for _ in reference_angles]

    return reference_positions


def rotate_reference_positions(
    reference_positions: List[Tuple[Quantity, Quantity]],
    n_steps: int,
) -> List[List[Tuple[Quantity, Quantity]]]:
    """
    Rotate a given list of `reference_positions`.

    The positions returned by `get_reference_positions()` are somewhat
    arbitrary, as they depend, for example, on an arbitrary choice of
    the exclusion angle. Preliminary experiments have shown that the
    exact placement of the reference apertures can have an unreasonably
    large effect on the SNR; meaning that a already slightly different
    (and arguably just as valid) choice of the reference positions can
    result in a significantly different SNR.
    For this reason, this function allows us to add some variation to
    the placement of the reference positions: we drop one reference
    position and then rotate the remaining positions over the opening
    angle of the dropped position / aperture. Using this function, we
    can compute multiple SNRs for a signal candidate, and average them
    or look at the standard deviation to get a feeling for how much
    the SNR depends on the exact placement of the reference apertures.

    Args:
        reference_positions: A list of reference positions, as returned
            by `get_reference_positions()`.
        n_steps: The number of rotation steps, that is, the number of
            rotated reference positions to be returned.

    Returns:
        A list where each element is a list containing one rotated
        version of the `reference_positions`.
    """

    # Ensure that we have at least 2 reference positions
    if len(reference_positions) < 2:
        raise RuntimeError(
            'Need at least two reference positions to rotate them!'
        )

    # Determine opening angle by computing the difference in position angle
    # between the first two reference positions.
    opening_angle = abs(reference_positions[1][1] - reference_positions[0][1])

    # Loop over offsets and compute rotated reference positions
    rotated_reference_positions = []
    for offset in np.linspace(Quantity(0, 'degree'), opening_angle, n_steps):
        rotated_reference_positions.append(
            [(_, __ + offset) for _, __ in reference_positions[:-1]]
        )

    return rotated_reference_positions
