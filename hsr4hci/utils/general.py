"""
General purpose utilities, e.g., cropping arrays.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from math import modf
from typing import List, Sequence, Tuple, Union

from astropy.nddata.utils import add_array
from scipy import ndimage

import numpy as np


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def add_array_with_interpolation(array_large: np.ndarray,
                                 array_small: np.ndarray,
                                 position: Tuple[Union[int, float],
                                                 Union[int, float]]
                                 ) -> np.ndarray:
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


def split_into_n_chunks(sequence: Sequence,
                        n_chunks: int) -> List[Sequence]:
    """
    Split a given `sequence` (list, numpy array, ...) into `n_chunks`
    "chunks" (sub-sequences) of approximately equal length, which are
    returned as a list.

    Source: https://stackoverflow.com/a/2135920/4100721

    Args:
        sequence: The sequence which should be split into chunks.
        n_chunks: The number of chunks into which `sequence` is split.

    Returns:
        A list containing the original sequence split into chunks.
    """

    # Basic sanity checks
    if len(sequence) < n_chunks:
        raise ValueError(f"n_chunks is greater than len(sequence): "
                         f"{n_chunks} > {len(sequence)}")
    if n_chunks <= 0:
        raise ValueError("n must be a positive integer!")

    # Compute number of chunks (k) and number of chunks with extra elements (m)
    k, m = divmod(len(sequence), n_chunks)

    # Split the sequence into n chunks of (approximately) size k
    return [sequence[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
            for i in range(n_chunks)]
