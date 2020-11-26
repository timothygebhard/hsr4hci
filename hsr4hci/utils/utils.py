"""
Utility functions, mostly for validating function arguments.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any

import numpy as np


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def check_consistent_size(*arrays: np.ndarray, axis: int = 0) -> bool:
    """
    Check that all `arrays` have a consistent size along the given
    `axis`. Can be used, for example, to ensure that a given `stack`
    and `parang` have the same number of frames.

    Args:
        *arrays: The (numpy) arrays that will be checked.
        axis: The axis along which to check for consistent sizes. Per
            default, this is the temporal axis (axis = 0).

    Returns:
        None if all `arrays` have consistent sizes; otherwise, a
        ValueError is raised.
    """

    # Make sure that all our arguments are actually arrays
    if not all(isinstance(_, np.ndarray) for _ in arrays):
        raise TypeError('All arguments must be numpy arrays!')

    # Get the sizes along the target axis
    sizes = [_.shape[axis] for _ in arrays]

    # Check if all sizes are the same (i.e., they are consistent)
    uniques = np.unique(sizes)
    if len(uniques) > 1:
        raise ValueError('At least two inputs have inconsistent sizes!')

    return True


def check_frame_size(frame_size: Any) -> bool:
    """
    Check if a given `frame_size` tuple represents a valid frame size
    (i.e., is a 2-tuple of positive integers).

    Args:
        frame_size: Variable which to check whether or not it represents
            a valid frame size.

    Returns:
        None if `frame_size` is a valid frame size; otherwise, a
        ValueError is raised.
    """

    if not (
        isinstance(frame_size, (tuple, list, np.ndarray))
        and len(frame_size) == 2
        and all(isinstance(_, int) for _ in frame_size)
        and all(_ > 0 for _ in frame_size)
    ):
        raise ValueError('frame_size is not a valid frame size!')

    return True


def check_cartesian_position(
    position: Any,
    require_int: bool = False,
) -> bool:
    """
    Check if a given `position` represents a valid Cartesian position
    (i.e., is a 2-tuple of floats or integers).

    Args:
        position: Variable to check whether it is a valid position.
        require_int: Whether or not we require all entries of the
            position to be integers (e.g., because we want to use
            `position` to index an array).

    Returns:
        None is `position` is a valid Cartesian position; otherwise, a
        ValueError is raised.
    """

    # Ensure that position is a 2-tuple of integers or floats
    if not (
        isinstance(position, (tuple, list, np.ndarray))
        and len(position) == 2
        and all(isinstance(_, (int, float)) for _ in position)
    ):
        raise ValueError('position is not a valid Cartesian position!')

    # If requested, make sure all entries of position are actually integers
    if require_int and not all(isinstance(_, int) for _ in position):
        raise ValueError('Not all entries of position are integers!')

    return True
