"""
General purpose utilities, e.g., cropping arrays.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

from astropy.nddata.utils import add_array
from math import modf
from scipy import ndimage

from typing import Tuple, Union


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------
def add_array_with_interpolation(array_large: np.ndarray,
                                 array_small: np.ndarray,
                                 position: Tuple[Union[int, float],
                                                 Union[int, float]]):
    """
    An extension of astropy.nddata.utils.add_array to add a smaller
    array at a given position in a larger array. In this version, the
    position may also be a float, in which case bilinear interpolation
    is used when adding array_small into array_large.

    Args:
        array_large: Large array, into which array_small is added into.
        array_small: Small array, which is added into array_large.
        position: The target position of the small array’s center, with
            respect to the large array. Coordinates should be in the
            same order as the array shape, but can also be floats.

    Returns:
        The new array, constructed as the the sum of `array_large`
        and `array_small`.
    """

    # Split the position into its integer and its fractional parts
    fractional_position, integer_position = \
        tuple(zip(*tuple(modf(x) for x in position)))

    # Create an empty with the same size as array_larger and add the
    # small array at the approximately correct position
    dummy = np.zeros_like(array_large)
    dummy = add_array(dummy, array_small, integer_position)

    # Use scipy.ndimage.shift to shift the array to the exact
    # position, using bilinear interpolation
    dummy = ndimage.shift(dummy, fractional_position, order=1)

    return array_large + dummy


def crop_center(array: np.ndarray,
                size: Tuple[int, ...]) -> np.ndarray:
    """
    Crop an n-dimensional array to the given size around its center.
    
    Args:
        array: The numpy array to be cropped.
        size: A tuple containing the size of the cropped array. To not
            crop along a specific axis, you can specify the size of
            that axis as -1.

    Returns:
        The input array, cropped to the desired size around its center.
    """

    start = tuple(map(lambda x, dx: None if dx == -1 else x//2 - dx//2,
                      array.shape, size))
    end = tuple(map(lambda x, dx: None if dx == -1 else x + dx, start, size))
    slices = tuple(map(slice, start, end))

    return array[slices]
