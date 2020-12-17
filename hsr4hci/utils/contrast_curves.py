"""
Utilities for computing contrast curves (under heavy development).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import List, Tuple, Union

from astropy.units import Quantity

import numpy as np

from hsr4hci.utils.apertures import get_aperture_positions


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def flux_ratio_to_magnitudes(
    flux_ratio: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Convert a contrast given as a flux ratio to magnitudes.

    Args:
        flux_ratio: The contrast as a flux ratio; either as a single
            float or as a numpy array of floats.

    Returns:
        The contrast(s) in magnitudes.
    """

    return -2.5 * np.log10(flux_ratio)


def magnitude_to_flux_ratio(
    magnitudes: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Convert a contrast in magnitudes back to a flux ratio.

    Args:
        magnitudes: The contrast in magnitudes; either as a single
            float or as a numpy array of floats.

    Returns:
        The contrast(s) as a flux ratio.
    """

    return 10 ** (-magnitudes / 2.5)


def azimuth_id_to_angle(azimuth_id: str) -> float:
    """
    Convert an azimuthal identifier (used for fake planet positions) to
    to the polar angle at which it (approximately) is injected.

    The azimuthal identifier are to be interpreted as follows: "a" is
    the position whose polar angle is the closest to 0°, "b" is the
    position whose polar angle is the closest to 60°, and so on. In
    practice, the injection positions are not always exactly at these
    angles; the exact angle depends on the separation.

    Args:
        azimuth_id: Azimuthal identifier: "a", "b", "c", "d", "e", "f".

    Returns:
        The polar angle that corresponds to this azimuthal identifier.
    """

    # Define lookup table
    lookup_table = {"a": 0, "b": 60, "c": 120, "d": 180, "e": 240, "f": 300}

    # Resolve given azimuth_id
    if azimuth_id not in lookup_table.keys():
        raise ValueError('Invalid injection_key: must be "a", "b", ..., "f"!')
    return lookup_table[azimuth_id]


def get_injection_and_reference_positions(
    separation: Quantity,
    azimuth_id: str,
    aperture_radius: Quantity,
    frame_size: Tuple[int, int],
) -> Tuple[Tuple[float, float], List[Tuple[float, float]]]:
    """
    This function computes the position at which a fake planet should
    be injected when estimating the throughput of a HCIpp algorithm,
    as well as the "reference positions", which are the positions of
    all other apertures that can be placed at the same separation.

    Args:
        separation: The separation from the center.
        azimuth_id: The azimuthal identifier of the fake planet. "a"
            refers to a planet at a polar angle of 0°, "b" to a planet
            at (approximately) 60° and so on (the exact angle depends
            on the separation).
        aperture_radius: The radius of the apertures; usually this is
            chosen as 0.5 lambda / D.
        frame_size: A tuple of integers `(width, height)` that specifies
            the size of the frames we are working with.

    Returns:
        A tuple `(injection_position, reference_positions)`.
        The former is a tuple of floats, containing the *Cartesian*
        position at which a fake planet should be injected.
        The latter is a list of tuples of floats, containing the
        Cartesian positions of all other apertures at this separation.
    """

    # Get the (default) apertures for this separation. This is simply a list
    # of the positions of the apertures that can be placed at this separation,
    # sorted by their polar angle.
    positions = get_aperture_positions(
        separation=separation,
        aperture_radius=aperture_radius,
        frame_size=frame_size,
    )

    # Find the index of the position whose angle is the closest to the "ideal"
    # angle for this azimuthal_id
    injection_position_idx = np.searchsorted(
        a=np.linspace(0, 360, len(positions), endpoint=False),
        v=azimuth_id_to_angle(azimuth_id),
        side='left',
    )

    # Split the positions into the injection and reference position(s)
    injection_position = positions[injection_position_idx]
    reference_positions = [
        _ for i, _ in enumerate(positions) if i != injection_position_idx
    ]

    return injection_position, reference_positions
