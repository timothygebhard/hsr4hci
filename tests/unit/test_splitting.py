"""
Tests for splitting.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import pytest

from hsr4hci.splitting import AlternatingSplit


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__alternating_split() -> None:
    """
    Test `hsr4hci.splitting.AlternatingSplit`.
    """

    # Test case 1:
    splitter = AlternatingSplit(n_splits=2)
    splits_list = list(splitter.split(X=np.arange(6)))
    expectation = [
        (np.array([1, 3, 5]), np.array([0, 2, 4])),
        (np.array([0, 2, 4]), np.array([1, 3, 5])),
    ]
    for i in range(2):
        assert np.all(splits_list[i][0] == expectation[i][0])
        assert np.all(splits_list[i][1] == expectation[i][1])

    # Test case 2:
    splitter = AlternatingSplit(n_splits=3)
    splits_list = list(splitter.split(X=np.arange(6)))
    expectation = [
        (np.array([1, 2, 4, 5]), np.array([0, 3])),
        (np.array([0, 2, 3, 5]), np.array([1, 4])),
        (np.array([0, 1, 3, 4]), np.array([2, 5])),
    ]
    for i in range(3):
        assert np.all(splits_list[i][0] == expectation[i][0])
        assert np.all(splits_list[i][1] == expectation[i][1])

    # Test case 3:
    splitter = AlternatingSplit(n_splits=1)
    splits_list = list(splitter.split(X=np.arange(4)))
    expectation = [(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))]
    for i in range(1):
        assert np.all(splits_list[i][0] == expectation[i][0])
        assert np.all(splits_list[i][1] == expectation[i][1])

    # Test case 4:
    with pytest.raises(AssertionError) as error:
        _ = AlternatingSplit(n_splits=0)
    assert 'n_splits must be a positive integer!' in str(error)
