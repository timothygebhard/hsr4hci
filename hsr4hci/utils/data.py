"""
Functions and classes for loading and splitting data.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import h5py
import numpy as np

from typing import Tuple


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class TrainTestSplitter:

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
                             f'"even_odd", but is "{self.split_type}"!')

        # ---------------------------------------------------------------------
        # Save constructor arguments
        # ---------------------------------------------------------------------

        self.n_splits = n_splits
        self.split_type = split_type
        self.shuffle = shuffle
        self.random_seed = random_seed

    def split(self, stack):

        # ---------------------------------------------------------------------
        # Initialize indices
        # ---------------------------------------------------------------------

        # Get the size of the stack
        n_samples = stack.shape[0]

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


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def load_data(dataset_config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the dataset specified in the dataset_config.

    Args:
        dataset_config: A dictionary containing the part of an
            experiment config file which specifies the dataset.

    Returns:
        A tuple (stack, parang), containing numpy array with the frames
        and the parallactic angles.
    """

    # Define shortcuts
    file_path = dataset_config['file_path']
    stack_key = dataset_config['stack_key']
    parang_key = dataset_config['parang_key']
    subsample = dataset_config['subsample']

    # Read in the dataset and select subsample
    with h5py.File(file_path, 'r') as hdf_file:
        stack = np.array(hdf_file[stack_key][::subsample, ...])
        parang = np.array(hdf_file[parang_key][::subsample, ...])

    return stack, parang
