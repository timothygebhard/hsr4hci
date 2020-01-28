"""
Masking utilities.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from cmath import polar
from typing import List, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_positions_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Convert a numpy mask into a list of positions selected by that mask.

    Args:
        mask: A numpy array containing only boolean values (or values
            that can be interpreted as such).

    Returns:
        A sorted list of all positions (x, y) with mask[x, y] == True.
    """
    return sorted(list((x, y) for x, y in zip(*np.where(mask))))


def get_circle_mask(mask_size: tuple,
                    radius: float,
                    center: tuple = None) -> np.ndarray:
    """
    Create a circle mask of a given size.

    Args:
        mask_size: A tuple (width, height) containing the size of the
            mask (in pixels) to be created. Should match the size of
            the array which is masked.
        radius: Radius of the disk in pixels.
        center: Center of the circle. If None is given, the disk
            will be centered within the array. This is the default.
    Returns:
        A numpy array of the specified size which is zero everywhere,
        except in a circular region of given radius around the
        specified disk center.
    """

    x_size, y_size = mask_size
    x, y = np.ogrid[:x_size, :y_size]

    if center is None:
        x_offset, y_offset = int(x_size / 2), int(x_size / 2)
        center = (x_offset, y_offset)

    circle_mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2 < radius ** 2)

    return circle_mask


def get_annulus_mask(mask_size: tuple,
                     inner_radius: float,
                     outer_radius: float) -> np.ndarray:
    """
    Create an annulus-shaped mask.

    Args:
        mask_size: A tuple (width, height) containing the size of the
            mask (in pixels) to be created. Should match the size of
            the array which is masked.
        inner_radius: Inner radius (in pixels) of the annulus mask.
        outer_radius: Outer radius (in pixels) of the annulus mask.

    Returns:
        A 2D numpy array of size `mask_size` which masks an annulus
        with a given `inner_radius` and `outer_radius`.
    """

    return np.logical_xor(get_circle_mask(mask_size=mask_size,
                                          radius=inner_radius),
                          get_circle_mask(mask_size=mask_size,
                                          radius=outer_radius))


def get_wedge_mask(mask_size: Tuple[int, int],
                   orientation_angle: float,
                   opening_angle: float) -> np.ndarray:
    """
    Create a wedge-shaped (i.e., a circle sector) mask of a given size.
    The wedge is always centered at the center of the mask.

    Args:
        mask_size: A tuple (width, height) specifying the size of the
            mask to be created.
        orientation_angle: The angle which defines the orientation of
            the wedge (in degrees). The orientation is given by the
            angle bisector (see below).
        opening_angle: The opening angle (in degrees); the wedge will
            mask all pixels in the sector:
                [orientation_angle - opening_angle / 2,
                 orientation_angle + opening_angle / 2]

    Returns:
        A numpy array containing a wedge-shaped binary mask.
    """

    # Convert angles from degree to radian (and use shorter names)
    theta = np.deg2rad(orientation_angle)
    phi = np.deg2rad(opening_angle)

    # Create a suitable grid
    x_, y_ = np.ogrid[:mask_size[0], :mask_size[1]]
    x = x_ - mask_size[0] / 2
    y = y_ - mask_size[1] / 2

    # Catch two quick corner cases: All pixels or no pixels
    if opening_angle == 0:
        return np.zeros(mask_size)
    if opening_angle == 360:
        return np.ones(mask_size)

    # The wedge is defined by two straight lines. To see if a point on the
    # grid belongs to the wedge, we need to know on which side of the lines
    # it falls, which we can find out by taking the scalar product with the
    # normal vector of said straight lines.
    scalarproduct_v = \
        (x * np.cos(theta + phi / 2.0) - y * np.sin(theta + phi / 2.0))
    scalarproduct_w = \
        (x * np.cos(theta - phi / 2.0) - y * np.sin(theta - phi / 2.0))

    # Ultimately, we only need to combine the two sets (i.e., half planes).
    # The way we do this depends whether the opening angle is smaller or
    # greater than 180 degrees.
    if opening_angle < 180:
        wedge_mask = np.logical_and(scalarproduct_v <= 0, scalarproduct_w >= 0)
    else:
        wedge_mask = np.logical_or(scalarproduct_v <= 0, scalarproduct_w >= 0)

    return wedge_mask


def get_sausage_mask(mask_size: Tuple[int, int],
                     position: Tuple[int, int],
                     radius: float,
                     opening_angle: float) -> np.ndarray:
    """
    Get a "sausage"-shaped mask, which corresponds to the shape you get
    if you place a circle with `sausage_radius` at the given `position`
    and then slide it along an arc with the given `opening_angle` around
    the center of the mask (the `position` bisects this arc).

    This type of mask can be used to get the "exclusion region" of
    pixels that can not be used a predictors for a given pixel (because
    they may contain planet signal if the pixel at `position` contains
    planet signal). In this case, the opening angle should be *twice*
    the field rotation of the data set.

    Args:
        mask_size: A tuple (width, height) specifying the size of the
            mask to be created.
        position: A tuple (x, y) specifying the position which defines
            the sausage mask (i.e., the center of the "sausage").
        radius: The radius (in pixels) which defines the width of the
            "sausage".
        opening_angle: The angle (in degrees) which defines the length
            of the "sausage".

    Returns:
        A numpy array containing a "sausage"-shaped binary mask.
    """

    # Compute the center of the mask
    center = tuple([_ / 2 for _ in mask_size])

    # Convert the given position to polar coordinates
    r, phi = polar(complex(position[1] - center[1],
                           position[0] - center[0]))

    # Get star and end position in cartesian coordinates
    start_position = r * np.exp(1j * (phi - np.deg2rad(opening_angle / 2)))
    start_position = (np.imag(start_position) + center[0],
                      np.real(start_position) + center[1])
    end_position = r * np.exp(1j * (phi + np.deg2rad(opening_angle / 2)))
    end_position = (np.imag(end_position) + center[0],
                    np.real(end_position) + center[1])

    # Get the annulus mask
    annulus_mask = get_annulus_mask(mask_size=mask_size,
                                    inner_radius=(r - radius),
                                    outer_radius=(r + radius))

    # Get the wedge mask
    wedge_mask = get_wedge_mask(mask_size=mask_size,
                                orientation_angle=np.rad2deg(phi),
                                opening_angle=opening_angle)

    # Get "end cap" masks
    end_cap_mask_1 = get_circle_mask(mask_size=mask_size,
                                     radius=radius,
                                     center=start_position)
    end_cap_mask_2 = get_circle_mask(mask_size=mask_size,
                                     radius=radius,
                                     center=end_position)

    # Compute the final mask by intersecting the annulus mask with the wedge
    # mask and then adding the "end cap" masks.
    return np.logical_or(np.logical_and(annulus_mask, wedge_mask),
                         np.logical_or(end_cap_mask_1, end_cap_mask_2))
