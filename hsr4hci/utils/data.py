"""
Utility functions for loading data.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Optional, Tuple

import h5py
import numpy as np

from hsr4hci.utils.general import crop_center


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def load_data(file_path: str,
              stack_key: str = '/stack',
              parang_key: str = '/parang',
              psf_template_key: Optional[str] = None,
              frame_size: Optional[Tuple[int, int]] = None,
              presubtract: Optional[str] = None,
              subsample: int = 1,
              **_: Any) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load a dataset from the HDF5 file at the given `file_path`.

    Args:
        file_path: A string containing the path to the HDF5 file
            containing the data set to be loading (consisting of the
            stack of images, an array of parallactic angles, and
            optionally an unsaturated PSF template).
        stack_key: The key of the dataset in the HDF file that contains
            the stack, which is expected to be a 3D array with the
            following shape: (n_frames, width, height).
        parang_key: The key of the dataset in the HDF file that contains
            the parallactic angles, which is expected to be an 1D array
            with the following shape: (n_frames, )
        psf_template_key: Optionally, the key key of the dataset in the
            HDF file that contains the unsaturated PSF template, which
            is expected to be a 2D array of shape (width, height). The
            size does not have to match the spatial size of the stack.
        frame_size: A tuple (width, height) of integers specifying the
            spatial size (in pixels) to which the stack will be cropped
            around the center. Dimensions should be odd numbers. If None
            is given, the frames are not cropped.
        presubtract: If this parameter is set to "mean" or "median",
            we subtract the mean (or median) along the time axis from
            the stack before returning it.
        subsample: An integer specifying the subsampling factor for the
            stack. If set to n, we only keep every n-th frame. By
            default, all frames are kept (i.e., subsample=1).
        **_: Additional keywords (which will be ignored).

    Returns:
        A tuple (stack, parang, psf_template), containing numpy array
        with the frames, the parallactic angles and the unsaturated
        PSF template.
    """

    # Read in the dataset from the HDf file
    with h5py.File(file_path, 'r') as hdf_file:

        # Select stack and parallactic angles and subsample as desired
        stack = np.array(hdf_file[stack_key][::subsample, ...])
        parang = np.array(hdf_file[parang_key][::subsample, ...])

        # If applicable, also select the PSF template
        psf_template = None
        if psf_template_key is not None:
            psf_template = np.array(hdf_file[psf_template_key]).squeeze()

    # Spatially crop the stack around the center to the desired frame size
    if frame_size is not None:
        stack = crop_center(stack, (-1, frame_size[0], frame_size[1]))

    # If desired, pre-subtract mean or median from the stack
    if presubtract == 'median':
        stack -= np.nanmedian(stack, axis=0)
    elif presubtract == 'mean':
        stack -= np.nanmean(stack, axis=0)

    return stack, parang, psf_template
