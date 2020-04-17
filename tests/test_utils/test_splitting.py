"""
Tests for splitting.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

from hsr4hci.utils.splitting import TrainTestSplitter


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__split():

    # Test case 1:
    splitter = TrainTestSplitter(n_splits=2, split_type='k_fold')
    splits_list = list(splitter.split(n_samples=6))
    expectation = [(np.array([3, 4, 5]), np.array([0, 1, 2])),
                   (np.array([0, 1, 2]), np.array([3, 4, 5]))]
    for i in range(2):
        assert np.all(splits_list[i][0] == expectation[i][0])
        assert np.all(splits_list[i][1] == expectation[i][1])

    # Test case 2:
    splitter = TrainTestSplitter(n_splits=2, split_type='even_odd')
    splits_list = list(splitter.split(n_samples=6))
    expectation = [(np.array([1, 3, 5]), np.array([0, 2, 4])),
                   (np.array([0, 2, 4]), np.array([1, 3, 5]))]
    for i in range(2):
        assert np.all(splits_list[i][0] == expectation[i][0])
        assert np.all(splits_list[i][1] == expectation[i][1])

    # Test case 3:
    splitter = TrainTestSplitter(n_splits=3, split_type='even_odd')
    splits_list = list(splitter.split(n_samples=6))
    expectation = [(np.array([1, 2, 4, 5]), np.array([0, 3])),
                   (np.array([0, 2, 3, 5]), np.array([1, 4])),
                   (np.array([0, 1, 3, 4]), np.array([2, 5]))]
    for i in range(3):
        assert np.all(splits_list[i][0] == expectation[i][0])
        assert np.all(splits_list[i][1] == expectation[i][1])

    # Test case 4:
    splitter = TrainTestSplitter(n_splits=1, split_type='even_odd')
    splits_list = list(splitter.split(n_samples=4))
    expectation = [(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))]
    for i in range(1):
        assert np.all(splits_list[i][0] == expectation[i][0])
        assert np.all(splits_list[i][1] == expectation[i][1])

    # Test case 5:
    splitter = TrainTestSplitter(n_splits=1, split_type='k_fold')
    splits_list = list(splitter.split(n_samples=3))
    expectation = [(np.array([0, 1, 2]), np.array([0, 1, 2]))]
    for i in range(1):
        assert np.all(splits_list[i][0] == expectation[i][0])
        assert np.all(splits_list[i][1] == expectation[i][1])
