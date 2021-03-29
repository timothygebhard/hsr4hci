"""
Utilities related to apertures: numbers, positions, photometry.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import List, Tuple, Union

import math

from astropy.units import Quantity
from photutils import aperture_photometry, CircularAperture

import numpy as np

from hsr4hci.coordinates import get_center


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_number_of_apertures(
    separation: Quantity,
    aperture_radius: Quantity,
    exact: bool = True,
) -> int:
    """
    Compute the number of non-overlapping apertures with a given
    `aperture_radius` that can be placed at the given `separation`.

    Args:
        separation: The separation at which the apertures are to be
            placed (e.g., 1 lambda / D).
        aperture_radius: The radius of the apertures to be placed.
        exact: Whether to use the exact formula for the number of
            apertures, or an approximation. The latter is generally
            very good; there are only very few cases where the
            approximation over-estimates the number of apertures by 1.

    Returns:
        The number of apertures at the given separation as an integer.
    """

    # Convert the separation and the aperture radius to units of pixels (any
    # unit is fine here actually, as long as it is the same for both)
    big_r = separation.to('pixel').value
    small_r = aperture_radius.to('pixel').value

    # Sanity check: for too small separations, there are no non-overlapping
    # apertures; hence, we raise a ValueError.
    if small_r > big_r:
        raise ValueError(
            'The aperture_size must not be greater than the separation!'
        )

    # For the exact number of apertures at a given separation we need to use
    # the formula derived here: https://stackoverflow.com/a/56008236/4100721
    # Note: The additional round() call is necessary to mitigate issues due to
    # floating pointing precision. Without it, we sometimes get results like
    # "5.999999999999 apertures", which gets floored to 5 without the round().
    if exact:
        return int(math.floor(round(math.pi / math.asin(small_r / big_r), 3)))

    # Alternatively, we can use the following approximation from the Mawet et
    # al. (2014) paper, which is slightly faster to compute:
    return int(math.floor(math.pi * big_r / small_r))


def get_aperture_positions(
    separation: Quantity,
    aperture_radius: Quantity,
    frame_size: Tuple[int, int],
) -> List[Tuple[float, float]]:
    """
    Get the "default" positions of the apertures for a given separation.

    Args:
        separation: The separation at which the apertures are to be
            placed (e.g., 1 lambda / D).
        aperture_radius: Radius of the apertures to be placed.
        frame_size: A tuple of integers `(width, height)` specifying the
            size of the frames that we are working with. (Necessary to
            compute the (Cartesian) coordinates of the apertures.)

    Returns:
        A list of tuples `(x, y)` containing the positions of the
        apertures for the given `separation`.
    """

    # Compute frame center
    center = get_center(frame_size=frame_size)

    # Get number of apertures
    n_apertures = get_number_of_apertures(
        separation=separation,
        aperture_radius=aperture_radius,
    )

    # Get angles at which to place the apertures
    angles = np.linspace(0, 2 * np.pi, n_apertures, endpoint=False)

    # Compute x- and y-positions in Cartesian coordinates
    x_positions = separation.to('pixel').value * np.cos(angles) + center[0]
    y_positions = separation.to('pixel').value * np.sin(angles) + center[1]

    # Convert positions to a list of tuples
    positions = [
        (float(x), float(y)) for x, y in zip(x_positions, y_positions)
    ]

    return positions


def get_aperture_flux(
    frame: np.ndarray,
    position: Union[Tuple[float, float], List[Tuple[float, float]]],
    aperture_radius: Quantity,
) -> Union[float, np.ndarray]:
    """
    Get the integrated flux in an aperture (or multiple apertures) of
    the given size (i.e., `aperture_radius`) at the given `position(s)`.

    This function is essentially a convenience wrapper that bundles
    together `CircularAperture` and `aperture_photometry()`.

    Args:
        frame: A 2D numpy array of shape `(width, height)` containing
            the data on which to run the aperture photometry.
        position: A tuple `(x, y)` (or a list of such tuples) specifying
            the position(s) at which we place the aperture(s) on the
            `frame`.
        aperture_radius: The radius of the aperture. In case multiple
            positions are given, the same radius will be used for all
            of them.

    Returns:
        The integrated flux (= sum of all pixels) in the aperture.
    """

    # Create an aperture (or a set of apertures) at the given `position`
    aperture = CircularAperture(
        positions=position, r=aperture_radius.to('pixel').value
    )

    # Perform the aperture photometry, select the results
    photometry_table = aperture_photometry(frame, aperture, method='exact')
    results = np.array(photometry_table['aperture_sum'].data)

    # If there is only a single result (i.e., only a single position), we
    # can cast the result to float
    if len(results) == 1:
        return float(results)
    return results


def get_reference_aperture_positions(
    frame_size: Tuple[int, int],
    position: Tuple[float, float],
    aperture_radius: Quantity,
    ignore_neighbors: int = 1,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Compute the positions of the reference apertures (i.e., the
    apertures used to estimate the noise level) for a signal aperture
    that is placed at `position`.

    Args:
        frame_size: A tuple of integers `(width, height)` specifying the
            size of the frames that we are working with. (Necessary to
            compute the (Cartesian) coordinates of the apertures.)
        position: A tuple `(x, y)` specifying the position at which the
            signal aperture will be placed. Usually, this is the
            suspected of a planet.
        aperture_radius: The radius of the apertures to be placed.
        ignore_neighbors: The number of neighboring apertures that will
            *not* be used as reference positions.
            Rationale: methods like PCA often cause visible negative
            self-subtraction "wings" left and right of the planet
            signal. As these do not provide an unbiased estimate of the
            background noise, we usually want to exclude them from the
            reference positions.

    Returns:
        A tuple `(reference, ignored)` where both `reference` and
        `ignored` are a list of tuples `(x, y)` with the positions of
        the actual reference apertures as well as the positions of the
        apertures that are ignored because of `ignore_neighbors`. If
        the latter is 0, the `ignored` list will be empty.
    """

    # Compute the frame center and polar representation of initial position
    center = get_center(frame_size=frame_size)
    rho = math.sqrt(
        (position[0] - center[0]) ** 2 + (position[1] - center[1]) ** 2
    )
    phi = math.atan2(position[1] - center[1], position[0] - center[0])

    # Compute the total number of apertures that can be placed at the
    # separation of the given `position`.This also includes the signal
    # aperture at `position`, which is why we will need to subtract 1.
    n_apertures = get_number_of_apertures(
        separation=Quantity(rho, 'pixel'), aperture_radius=aperture_radius
    )
    n_apertures -= 1

    # Collect the indices of the positions which are ignored (namely, the
    # first and last `ignore_neighbors`)
    ignored_idx = np.full(n_apertures, False)
    ignored_idx[:ignore_neighbors] = True
    ignored_idx = np.logical_or(ignored_idx, ignored_idx[::-1])

    # Get angles at which to place the apertures. The `+ 1` is needed to take
    # into account the position of the signal aperture.
    angles = np.linspace(0, 2 * np.pi, n_apertures + 1, endpoint=False)[1:]
    angles = np.mod(angles + phi, 2 * np.pi)

    # Compute x- and y-positions in Cartesian coordinates
    x_positions = rho * np.cos(angles) + center[0]
    y_positions = rho * np.sin(angles) + center[1]

    # Split the positions into reference and ignored positions and convert
    # them into lists of tuples
    reference = [
        (float(x), float(y))
        for x, y in zip(x_positions[~ignored_idx], y_positions[~ignored_idx])
    ]
    ignored = [
        (float(x), float(y))
        for x, y in zip(x_positions[ignored_idx], y_positions[ignored_idx])
    ]

    return reference, ignored
