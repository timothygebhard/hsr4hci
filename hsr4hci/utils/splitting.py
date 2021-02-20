"""
Utility functions and classes for performing train / test splits.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Iterator, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class AlternatingSplit:
    """
    Alternating split cross-validator.

    Provides train / test indices to split data in train / test sets.

    The split is performed in an "alternating" way:
    Assume that `n_splits=3`. In this case, the samples / data points
    are labeled: A B C A B C A B C ... In the first split, all points
    labeled A or B constitute the training set, and C is the test (or
    hold-out) set. In the second split, all points labeled A or C are
    used for training and B is the test split. In the final split, A
    is held out and training is performed on B and C.

    In the special case of `n_splits=2`, this is "even-odd splitting",
    that is, in the first split the points with even indices (in the
    data matrix X) are used for training (and odd indices are used for
    testing), and vice versa in the second split.

    This splitting scheme is useful for HCI/ADI data, because it means
    that the effective field rotation in all splits is the same (using
    standard k-fold splitting would---for k=2---cut the field rotation
    in the training data in half).
    """

    def __init__(self, n_splits: int) -> None:

        # Sanity check: we cannot have less than 1 split
        assert n_splits >= 1, 'n_splits must be a positive integer!'
        self.n_splits = n_splits

    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.

        Args:
            X: A 2D numpy array of shape (n_samples, n_features) that
                contains the training data.

        Yields:
            train_idx: A 1D numpy array containing the training set
                indices for that split.
            test_idx: A 1D numpy array containing the testing set
                indices for that split.
        """

        # Get the number of samples (= number of rows in X)
        n_samples = X.shape[0]

        # Initialize the array of indices that gets split into train / test
        indices = np.arange(n_samples)

        # If n_splits = 1, we do not need to split. Instead, we simply return
        # the indices right away such that train_idx == test_idx. (This is for
        # compatibility reasons in cases where we do not really want to split
        # the data into training and test.)
        if self.n_splits == 1:
            yield indices, indices
            return

        # Otherwise, generate indices for alternating splitting scheme
        for i in range(self.n_splits):

            test_idx = indices[i::self.n_splits]
            train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)

            yield train_idx, test_idx
