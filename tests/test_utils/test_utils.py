"""
Tests for splitting.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import pytest

from hsr4hci.utils.utils import check_consistent_size, check_frame_size


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------


def test__check_consistent_size() -> None:

    # Check that only numpy arrays are accepted
    with pytest.raises(TypeError) as type_error:
        # noinspection PyTypeChecker
        check_consistent_size(None)
    assert 'All arguments must be numpy arrays!' in str(type_error)

    # Case 1: Ensure the correct default case is accepted
    assert check_consistent_size(np.empty((5, 2)), np.empty((5, 3)))

    # Case 2: Ensure a correct non-default case is accepted
    assert check_consistent_size(np.empty((5, 3)), np.empty((4, 3)), axis=1)

    # Case 3: Consistency along first axis (default)
    with pytest.raises(ValueError) as value_error:
        check_consistent_size(np.empty((1, 5)), np.empty((2, 5)))
    assert 'At least two inputs have inconsistent sizes!' in str(value_error)

    # Case 4: Consistency along target axis
    with pytest.raises(ValueError) as value_error:
        check_consistent_size(np.empty((5, 2)), np.empty((5, 3)), axis=1)
    assert 'At least two inputs have inconsistent sizes!' in str(value_error)


def test__check_frame_size() -> None:

    # Case 1: frame_size is not a tuple, list or np.ndarray
    frame_size_1 = '10_20'
    with pytest.raises(ValueError) as value_error:
        check_frame_size(frame_size_1)
    assert 'frame_size is not a valid frame size!' in str(value_error)

    # Case 2: an entry of frame_size is not an integer
    frame_size_2 = (1.0, 1)
    with pytest.raises(ValueError) as value_error:
        check_frame_size(frame_size_2)
    assert 'frame_size is not a valid frame size!' in str(value_error)

    # Case 3: an entry of frame_size is not positive
    frame_size_3 = (1, 0)
    with pytest.raises(ValueError) as value_error:
        check_frame_size(frame_size_3)
    assert 'frame_size is not a valid frame size!' in str(value_error)
