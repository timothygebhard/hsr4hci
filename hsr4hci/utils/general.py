"""
General purpose utilities, e.g., cropping arrays.
"""
# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

from typing import Tuple


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

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
