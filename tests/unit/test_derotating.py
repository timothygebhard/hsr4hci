"""
Tests for derotating.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import pytest
import numpy as np

from hsr4hci.derotating import derotate_combine, derotate_frames


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__derotate_frames() -> None:

    # Case 1: no rotation at all
    stack = np.random.normal(0, 1, (10, 13, 13))
    parang = np.zeros(10)
    derotated = derotate_frames(stack=stack, parang=parang)
    assert np.allclose(stack, derotated)

    # Case 2: single frame, 90 degree rotation
    stack = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]])
    parang = np.array([90])
    derotated = derotate_frames(stack=stack, parang=parang)
    assert np.allclose(np.rot90(stack.squeeze(), k=-1), derotated.squeeze())

    # Case 3: parallel and serial derotation should return same result
    stack = np.random.normal(0, 1, (10, 13, 13))
    parang = np.random.normal(0, 2 * np.pi, 10)
    derotated_1 = derotate_frames(stack=stack, parang=parang, n_processes=1)
    derotated_2 = derotate_frames(stack=stack, parang=parang, n_processes=4)
    assert np.allclose(derotated_1, derotated_2)

    # Case 4: applying a mask to derotated frames
    stack = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]]).astype(float)
    parang = np.array([0])
    mask = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]]).astype(bool)
    derotated = derotate_frames(stack=stack, parang=parang, mask=mask)
    assert np.isnan(derotated[:, mask]).all()


def test__derotate_combine() -> None:

    # Case 1: no rotation at all should simply give mean
    stack = np.random.normal(0, 1, (10, 13, 13))
    parang = np.zeros(10)
    combined = derotate_combine(stack=stack, parang=parang, combine='mean')
    assert np.allclose(np.mean(stack, axis=0), combined)

    # Case 2: no rotation at all should simply give median
    stack = np.random.normal(0, 1, (10, 13, 13))
    parang = np.zeros(10)
    combined = derotate_combine(stack=stack, parang=parang, combine='median')
    assert np.allclose(np.median(stack, axis=0), combined)

    # Case 3: Illegal value for combine
    with pytest.raises(ValueError) as value_error:
        derotate_combine(stack=stack, parang=parang, combine='illegal')
    assert 'Illegal option for parameter "combine"!' in str(value_error)

    # Case 4: applying a mask to combined frames
    stack = np.random.normal(0, 1, (10, 3, 3))
    parang = np.random.normal(0, 2 * np.pi, 10)
    mask = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]]).astype(bool)
    combined = derotate_combine(stack=stack, parang=parang, mask=mask)
    assert np.isnan(combined[mask]).all()
