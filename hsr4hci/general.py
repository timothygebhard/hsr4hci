"""
General purpose utilities, e.g., cropping arrays.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from bisect import bisect_left
from functools import reduce
from typing import Any, Callable, List, Sequence, Tuple, Union, TypeVar

import operator

from scipy.ndimage import shift

import numpy as np


# -----------------------------------------------------------------------------
# TYPE VARS
# -----------------------------------------------------------------------------

T = TypeVar('T', np.ndarray, Sequence)


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

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
    if not array.ndim == len(size):
        raise RuntimeError(
            'Length of size must match number of dimensions of array!'
        )

    # Loop over the the axes of the array to create slices
    slices = list()
    for old_len, new_len in zip(array.shape, size):

        # Compute start and end position for axis: if the new length is
        # greater the the current one, we do not crop.
        if new_len > old_len:
            start = None
            end = None

        # Otherwise we crop the same amount on both sides
        else:
            start = old_len // 2 - new_len // 2 if new_len != -1 else None
            end = start + new_len if start is not None else None

        # Create a slice object for axis
        slices.append(slice(start, end))

    return np.asarray(array[tuple(slices)])


def prestack_array(
    array: np.ndarray,
    stacking_factor: int,
    stacking_function: Callable = np.mean,
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
        mean to reduce the size of the ADI stack.

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


def find_closest(sequence: Sequence, value: Any) -> Tuple[int, Any]:
    """
    Given a sorted `sequence`, find the entry (and its index) in it
    that is the closest to the given `value`.

    Original source: https://stackoverflow.com/a/12141511/4100721

    Args:
        sequence: A sequence (i.e., a list, tuple or array).
        value: A numeric value (i.e., usually an int or float).

    Returns:
        A tuple `(index, value)` where `value` is the entry in
        `sequence` that is the closest to `value`, and `index`
        is its index: `sequence[index] == value`.
    """

    pos = bisect_left(sequence, value)

    if pos == 0:
        return 0, sequence[0]
    if pos == len(sequence):
        return len(sequence) - 1, sequence[-1]

    before = sequence[pos - 1]
    after = sequence[pos]

    if after - value < value - before:
        return pos, after
    return pos - 1, before


def pad_array_to_shape(
    array: np.ndarray,
    shape: Tuple[int, ...],
    **kwargs: Any,
) -> np.ndarray:
    """
    Pad a numpy array to a given target shape (unlike `np.pad`, which
    adds a given amount of padding). By default, zero-padding is used.

    Args:
        array: An n-dimensional numpy array.
        shape: The tuple of integers specifying the target shape to
            which the `array` is padded. The length of this tuple must
            match exactly the number of dimensions of `array`, i.e.,
            this function will *not* automatically add new axes (use
            `array.reshape()` to add a new dimension first for this).
            Also, the new `shape` must be greater or equal to the
            current `array.shape` for every axis, i.e., this function
            cannot be used for negative padding (cropping).
        kwargs: Additional keyword arguments that are passed to
            `np.pad()`; for example `constant_values` to determine
            the value with which the array is padded.

    Returns:
        A copy of the given `array` that has been padded to the given
        `shape`.
    """

    # Make sure that `array` and `shape` have matching dimensions
    if not array.ndim == len(shape):
        raise ValueError('Dimensions of array and shape do not align!')

    # Compute padding tuples
    pad_width = []
    for i in range(array.ndim):

        # Compute difference between target size and current size and ensure
        # that it is non-negative (this function does not handle cropping!)
        difference = float(shape[i] - array.shape[i])
        if difference < 0:
            raise ValueError(
                f'Target size {shape[i]} along axis {i} is smaller than '
                f'current size {array.shape[i]}!'
            )

        # If the difference is non-negative, compute the left and right padding
        padding = (int(difference // 2), int(difference // 2 + difference % 2))
        pad_width.append(padding)

    # Finally, call np.pad() with the pad_width values that we have computed
    return np.asarray(
        np.pad(array=array, pad_width=pad_width, mode='constant', **kwargs)
    )


def crop_or_pad(array: np.ndarray, size: Tuple[int, ...]) -> np.ndarray:
    """
    Take an `array` and a target `size` and either crop or pad the
    `array` to match the given size.

    Args:
        array: A numpy array.
        size: A tuple of integers specifying the target size.

    Returns:
        The original `array`, cropped or padded to match the `size`.
    """

    # If all array dimensions are larger than the target, we crop the array
    if all(array.shape[_] >= size[_] for _ in range(array.ndim)):
        return crop_center(array, size)

    # If all array dimensions are smaller than the target, we pad the array
    if all(array.shape[_] <= size[_] for _ in range(array.ndim)):
        return pad_array_to_shape(array, size)

    # If some dimensions are larger and some are smaller, we raise an error
    raise RuntimeError('Mixing of cropping and padding is not supported!')


def shift_image(
    image: np.ndarray,
    offset: Tuple[float, float],
    interpolation: str = 'bilinear',
    mode: str = 'constant',
) -> np.ndarray:
    """
    Function to shift a 2D array (i.e., an `image`) by a given `offset`.

    This function is essentially a simplified port of the PynPoint
    function of the same name (`pynpoint.util.image.shift_image()`).

    Args:
        image: A 2D numpy array containing the image to be shifted.
        offset: A tuple of floats `(x_shift, y_shift)` containing the
            amount (in pixels) how much the `image` should be shifted.
        interpolation: The interpolation method to be used. Must be one
            of the following: 'spline', 'bilinear'.
            Default is 'bilinear' because it is flux-preserving.
        mode: The mode parameter determines how the input array is
            extended beyond its boundaries. See `scipy.ndimage.shift()`
            for a full documentation.

    Returns:
        The `image` shifted by the amount specified in `offset`.
    """

    # Ensure that the image is really 2D
    if image.ndim != 2:
        raise ValueError('Input image must be 2D!')

    # Call the respective scipy routine with the correct arguments.
    # NOTE: We flip the order of `offset` here, because shift() operates on
    # a numpy array, which uses the matrix-like indexing convention, unlike
    # the rest of the code, which uses "intuitive" coordinates.
    if interpolation == 'spline':
        shifted_image = shift(image, offset[::-1], order=5, mode=mode)
    elif interpolation == 'bilinear':
        shifted_image = shift(image, offset[::-1], order=1, mode=mode)
    else:
        raise ValueError(
            'The value of interpolation must be one of the following: '
            '"spline", "bilinear"!'
        )

    return np.asarray(shifted_image)


def flatten_nested_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
    """
    Flatten a nested dictionary into a dictionary with only 1 level.

    Example: `flatten_nested_dict({'a': {'b': 5}, 'c': 2})` will produce
    the following output: `{'a_b': 5, 'c': 2}`.

    Original source: https://stackoverflow.com/a/6027615/4100721

    Args:
        d: Dictionary to be flattened.
        parent_key: Key of the parent of `d`; this is needed because
            the function calls itself recursively.
        sep: The separator to use for merging keys.

    Returns:
        A flattened version of the input dictionary.
    """
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
