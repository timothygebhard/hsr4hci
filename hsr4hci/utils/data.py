"""
Utility functions for loading data.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Optional, Tuple

import h5py
import numpy as np

from hsr4hci.utils.general import crop_center


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def load_data(dataset_config: dict) -> Tuple[np.ndarray, np.ndarray,
                                             Optional[np.ndarray]]:
    """
    Load the dataset specified in the dataset_config.

    Args:
        dataset_config: A dictionary containing the part of an
            experiment config file which specifies the dataset.

    Returns:
        A tuple (stack, parang, psf_template), containing numpy array
        with the frames, the parallactic angles and the unsaturated
        PSF template.
    """

    # Define some shortcuts
    file_path = dataset_config['file_path']
    stack_key = dataset_config['stack_key']
    parang_key = dataset_config['parang_key']
    psf_template_key = dataset_config['psf_template_key']
    frame_size = dataset_config['frame_size']
    subsample = dataset_config['subsample']
    presubtract = dataset_config['presubtract']

    # Read in the dataset from the HDf file
    with h5py.File(file_path, 'r') as hdf_file:

        # Select stack and parallactic angles
        stack = np.array(hdf_file[stack_key][::subsample, ...])
        parang = np.array(hdf_file[parang_key][::subsample, ...])

        # Spatially crop the stack to the desired frame size without
        # changing the number of frames in it
        stack = crop_center(stack, (-1, frame_size[0], frame_size[1]))

        # Pre-subtract mean or median from stack (if desired)
        if presubtract == 'median':
            stack -= np.nanmedian(stack, axis=0)
        elif presubtract == 'mean':
            stack -= np.nanmean(stack, axis=0)

        # If applicable, also select the PSF template
        if psf_template_key is not None:
            psf_template = np.array(hdf_file[psf_template_key]).squeeze()
        else:
            psf_template = None

    return stack, parang, psf_template
