"""
General purpose utilities, e.g., cropping arrays.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from bisect import bisect_left
from functools import reduce
from hashlib import md5
from math import modf
from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple, Union

import operator

from astropy.nddata.utils import add_array
from scipy import ndimage

import numpy as np


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def add_array_with_interpolation(
    array_large: np.ndarray,
    array_small: np.ndarray,
    position: Tuple[float, float],
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
    fractional_position, integer_position = tuple(
        zip(*tuple(modf(x) for x in position))
    )

    # Create an empty with the same size as array_larger and add the
    # small array at the approximately correct position
    dummy = np.zeros_like(array_large)
    dummy = add_array(dummy, array_small, integer_position)

    # Use scipy.ndimage.shift to shift the array to the exact
    # position, using bilinear interpolation
    dummy = ndimage.shift(dummy, fractional_position, order=1)

    return array_large + dummy


def crop_center(
    array: np.ndarray,
    size: Tuple[int, ...],
) -> np.ndarray:
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

    # Ensure that the the array shape and the size variable match
    assert array.ndim == len(
        size
    ), 'Length of size must match number of dimensions of array!'

    # Ensure that we are not "cropping" to something larger than the input
    for (in_size, out_size) in zip(array.shape, size):
        assert (
            out_size <= in_size or out_size == -1
        ), 'Output size cannot be larger than input size when cropping array!'

    # Loop over the the axes of the array to create slices
    slices = list()
    for old_len, new_len in zip(array.shape, size):

        # Compute start and end position for axis
        start = old_len // 2 - new_len // 2 if new_len != -1 else None
        end = start + new_len if start is not None else None

        # Create a slice object for axis
        slices.append(slice(start, end))

    return array[tuple(slices)]


def split_into_n_chunks(
    sequence: Sequence,
    n_chunks: int,
) -> List[Sequence]:
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
        raise ValueError(
            f"n_chunks is greater than len(sequence): "
            f"{n_chunks} > {len(sequence)}"
        )
    if n_chunks <= 0:
        raise ValueError("n_chunks must be a positive integer!")

    # Compute number of chunks (k) and number of chunks with extra elements (m)
    k, m = divmod(len(sequence), n_chunks)

    # Split the sequence into n chunks of (approximately) size k
    return [
        sequence[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
        for i in range(n_chunks)
    ]


def prestack_array(
    array: np.ndarray,
    stacking_factor: int,
    stacking_function: Callable = np.median,
    axis: int = 0,
) -> np.ndarray:
    """
    Perform "pre-stacking" on a given `array`: The array is split into
    blocks (each of size `stacking_factor`) along the given axis, and
    the given `stacking_function` is applied to each block (again along
    the specified axis). The results for each block are combined and
    returned, resulting in a numpy array that has the same shape as the
    input array, except that the specified axis has been reduced by the
    given stacking factor.

    Example use case: Replace each block of 50 raw frames by their
        median to reduce the size of the ADI stack.

    Args:
        array: A numpy array containing the input array.
        stacking_factor: An integer defining how many elements of the
            input go into one block and are combined for the output.
        stacking_function: The function to be used for combining the
            blocks. Usually, this will be `np.mean` or `np.median`.
        axis: Axis along which the stacking operation is performed. By
            default, we stack along the time axis, which by convention
            is the first axis (0).

    Returns:
        A version of the input `stack` where blocks of the specified
        size have been merged using the givens `stacking_function`.
    """

    # If stacking factor is 1, we do not need to stack anything
    if stacking_factor == 1:
        return array

    # Otherwise, we compute the number of splits and splitting indices
    n_splits = np.ceil(array.shape[0] / stacking_factor).astype(int)
    split_indices = [i * stacking_factor for i in range(1, n_splits)]

    # Now, split the input array into that many splits, merge each split
    # according to the given stacking_function, and return the result
    return np.stack(
        [
            stacking_function(block, axis=axis)
            for block in np.split(array, split_indices, axis=axis)
        ],
        axis=axis,
    )


def get_from_nested_dict(
    nested_dict: dict,
    location: Sequence,
) -> Any:
    """
    Get a value from a nested dictionary at a given location, described
    by a sequence of keys.

    Examples:
        >>> dictionary = {'a': {'b': 42}}
        >>> get_from_nested_dict(dictionary, ['a', 'b'])
        42

    Args:
        nested_dict: A nested dictionary.
        location: The location within the nested dictionary, described
            by a sequence (i.e., a list or tuple) of keys used to access
            the target value.

    Returns:
        The value of the `nested_dict` at the specified location.
    """

    return reduce(operator.getitem, location, nested_dict)


def set_in_nested_dict(
    nested_dict: dict,
    location: Sequence,
    value: Any,
) -> None:
    """
    Set a value at a given location (described by a sequence of keys)
    in a nested dictionary.

    Examples:
        >>> dictionary = {'a': {'b': 42}}
        >>> set_in_nested_dict(dictionary, ['a', 'b'], 23)
        >>> dictionary
        {'a': {'b': 23}}

    Args:
        nested_dict: A nested dictionary.
        location: The target location within the nested dictionary,
            described by a sequence (i.e., a list or tuple) of keys
            used to access the target value.
        value: The value to be written to the target location.
    """

    get_from_nested_dict(nested_dict, location[:-1])[location[-1]] = value


def rotate_position(
    position: Union[Tuple[float, float], np.ndarray],
    center: Tuple[float, float],
    angle: Union[float, np.ndarray],
) -> Union[Tuple[float, float], np.ndarray]:
    """
    Take a `position` and rotate it around the given `center` for the
    given `angle`. Either the `position` or the `angle` can also be an
    array (but not both).

    Args:
        position: The initial position as a 2-tuple `(x, y)`, or as a
            numpy array of shape `(2, n_positions)`.
        center: The center of the rotation as a 2-tuple `(x_c, y_c)`.
        angle: The rotation angle *in degrees* (not radian); either as
            a float or as a numpy array of shape `(n_angles, )`.

    Returns:
        The rotated position, either as a 2-tuple, or as a numpy array
        of shape `(2, n_positions)`.
    """

    # Make sure that not both `position` and `angle` are arrays
    if isinstance(position, np.ndarray) and isinstance(angle, np.ndarray):
        raise ValueError('position and angle cannot both be arrays!')

    # Convert angle from degree to radian for numpy
    phi = np.deg2rad(angle)

    # Compute x- and y-coordinate of rotated position(s)
    x = (
        center[1]
        + (position[0] - center[1]) * np.cos(phi)
        - (position[1] - center[0]) * np.sin(phi)
    )
    y = (
        center[0]
        + (position[0] - center[1]) * np.sin(phi)
        + (position[1] - center[0]) * np.cos(phi)
    )

    # If we called the function on an array, we have to return an array
    if isinstance(x, np.ndarray):
        return np.vstack((x, y))
    return x, y


def get_md5_checksum(
    file_path: Union[Path, str],
    buffer_size: int = 8192,
) -> str:
    """
    Compute the MD5 checksum of the file at the given `file_path`.

    Args:
        file_path: The to the file of which we want to compute the MD5
            checksum.
        buffer_size: Buffer size (in bytes) for reading in the target
            file in chunks.

    Returns:
        The MD5 checksum of the specified file.
    """

    # Initialize the MD5 checksum
    md5_checksum = md5()

    # Open the input file and process it in chunks, updating the MD5 hash
    with open(file_path, 'rb') as binary_file:
        for chunk in iter(lambda: binary_file.read(buffer_size), b""):
            md5_checksum.update(chunk)

    return str(md5_checksum.hexdigest())


def find_closest(sequence: Sequence, value: Any) -> Any:
    """
    Given a sorted `sequence`, find the entry in it that is the closest
    to the given `value`.

    Source: https://stackoverflow.com/a/12141511/4100721

    Args:
        sequence: A sequence (basically: a list, tuple or array).
        value: A numeric value (i.e., usually an int or float).

    Returns:
        The entry in `sequence` that is the closest to `value`.
    """

    pos = bisect_left(sequence, value)

    if pos == 0:
        return sequence[0]
    if pos == len(sequence):
        return sequence[-1]

    before = sequence[pos - 1]
    after = sequence[pos]

    if after - value < value - before:
        return after
    return before


def fast_corrcoef(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    A fast(er) way to compute the correlation coefficient between
    the variables `x` and `y`, based on `numpy.einsum()`.

    For array sizes between 2 and 10^8, this implementation is, on
    average, 4.2 (+-2.2) times faster than `numpy.corrcoef()`, and
    2.9 (+-1.1) times faster than `scipy.stats.pearsonr()`.

    Args:
        x: A numpy array with realizations of the random variable X.
        y: A numpy array with realizations of the random variable Y.

    Returns:
        The correlation coefficient Corr(X, Y).
    """

    n = x.shape[0]
    dx = x - (np.einsum("n->", x) / np.double(n))
    dy = y - (np.einsum("n->", y) / np.double(n))

    cov = np.einsum("i,i->", dy, dx)

    var_x = np.einsum("n,n->", dy, dy)
    var_y = np.einsum("n,n->", dx, dx)
    tmp = var_x * var_y

    return float(cov / np.sqrt(tmp))
