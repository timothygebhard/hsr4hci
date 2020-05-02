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

class TrainTestSplitter:
    """
    A class that can be used for splitting a data set into train / test.

    Args:
        n_splits: Number of splits, essentially the "k" in k-fold cross
            validation (see also documentation of split() method).
        split_type: The split type, which must be either 'k_fold' or
            'even_odd'.
        shuffle: Whether or not to shuffle the data before splitting.
        random_seed: Seed for the random number generator used when
            `shuffle` is set to True.
    """

    def __init__(self,
                 n_splits: int = 2,
                 split_type: str = 'k_fold',
                 shuffle: bool = False,
                 random_seed: int = 42):

        # ---------------------------------------------------------------------
        # Perform some basic sanity checks
        # ---------------------------------------------------------------------

        if n_splits < 1:
            raise ValueError('n_splits must be greater or equal than 1!')

        if split_type not in ('k_fold', 'even_odd'):
            raise ValueError(f'split_type must be either "k_fold" or '
                             f'"even_odd", but is "{split_type}"!')

        # ---------------------------------------------------------------------
        # Save constructor arguments
        # ---------------------------------------------------------------------

        self.n_splits = n_splits
        self.split_type = split_type
        self.shuffle = shuffle
        self.random_seed = random_seed

    def split(self,
              n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Take the number of samples and return the indices for the split.

        Note that this function does not return, but instead yield its
        results such that we can loop over the results, as we would
        like to do, for example, for cross-validation.
        Example: Assume we are doing a two-fold split where we split our
        data into two parts A and B. In the first round, A will be the
        training set and B the testing set. In the second round, the
        roles are reversed and B is returned as the training set, while
        A is used as the testing set.

        Args:
            n_samples: The number of samples in the data set we want to
                split into training and test.

        Returns:
            A generator of tuples (train_indices, test_indices) which
            can be used to split the data set in the desired way.
        """

        # ---------------------------------------------------------------------
        # Initialize indices
        # ---------------------------------------------------------------------

        # Initialize the array of indices that gets split into train / test
        indices = np.arange(n_samples)

        # If desired, shuffle the indices
        if self.shuffle:
            rng = np.random.RandomState(seed=self.random_seed)
            rng.shuffle(indices)

        # ---------------------------------------------------------------------
        # Special case: n_splits = 1
        # ---------------------------------------------------------------------

        # If n_splits = 1, we do not need to split. Instead, we simply return
        # the indices right away such that train_idx == test_idx.
        if self.n_splits == 1:
            yield indices, indices
            return

        # ---------------------------------------------------------------------
        # Generate indices for k-fold splitting scheme
        # ---------------------------------------------------------------------

        if self.split_type == 'k_fold':

            # Create a dummy array containing the size of every fold
            fold_sizes = np.full(self.n_splits, n_samples // self.n_splits,
                                 dtype=np.int)
            fold_sizes[:n_samples % self.n_splits] += 1

            current = 0
            for fold_size in fold_sizes:

                test_idx = indices[current:(current + fold_size)]
                train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)

                yield train_idx, test_idx

                current += fold_size

        # ---------------------------------------------------------------------
        # Generate indices for even / odd splitting scheme
        # ---------------------------------------------------------------------

        elif self.split_type == 'even_odd':

            for i in range(self.n_splits):

                test_idx = indices[i::self.n_splits]
                train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)

                yield train_idx, test_idx

        # ---------------------------------------------------------------------
        # Otherwise, raise a ValueError
        # ---------------------------------------------------------------------

        else:
            raise ValueError()
