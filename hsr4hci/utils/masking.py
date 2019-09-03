"""
Masking utilities.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_positions_from_mask(mask: np.ndarray) -> list:
    """
    Convert a numpy mask into a list of positions selected by that mask.

    Args:
        mask: A numpy array containing only boolean values (or values
            that can be interpreted as such).

    Returns:
        A list of all positions (x, y) with mask[x, y] == True.
    """
    return list(zip(*np.where(mask)))


def get_circle_mask(mask_size: tuple,
                    radius: float,
                    center: tuple = None):
    """
    Create a circle mask of a given size.

    Args:
        mask_size: Size of the mask to be created. Should match the
            size of the array which is masked.
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
                     inner_exclusion_radius: float,
                     outer_exclusion_radius: float) -> np.ndarray:
    """
    Create a annulus-shaped mask.

    Args:
        mask_size:
        inner_exclusion_radius:
        outer_exclusion_radius:

    Returns:

    """

    return np.logical_xor(get_circle_mask(mask_size=mask_size,
                                          radius=inner_exclusion_radius),
                          get_circle_mask(mask_size=mask_size,
                                          radius=outer_exclusion_radius))
