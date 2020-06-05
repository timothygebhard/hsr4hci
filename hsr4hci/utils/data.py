"""
Utility functions for loading data.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Optional, Tuple

import h5py
import numpy as np

from hsr4hci.utils.config import get_data_dir
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


def load_default_data(
    planet: str,
    stacking_factor: int = 50,
    data_dir: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function is a convenience wrapper around ``load_data``, which
    allows to load our most common data sets with default settings.

    Args:
        planet: The name of the planet. Must be either "Beta_Pictoris",
            "HIP_65426", or "HR_8799".
        stacking_factor: The pre-stacking factor of the data set. In
            general, this must be in (1, 5, 10, 25, 50, 100).
        data_dir: The path to be base of the data directory, in case
            you do not want to use the system default defined in the
            environmental variable HSR4HCI_DATA_DIR.

    Returns:
        numpy arrays containing both the stack and the parallactic
        angles of the the requested data set.
    """

    # Define the data directory: either we are explicitly passed one, or
    # we use the default from the environmental variable HSR4HCI_DATA_DIR
    data_dir = data_dir if data_dir is not None else get_data_dir()

    # Hard-code some information about our most common data sets
    if planet == 'Beta_Pictoris':
        planet_part = ('Beta_Pictoris', 'Lp', '2013-02-01')
        frame_size = (81, 81)
    elif planet == 'HIP_65426':
        planet_part = ('HIP_65426', 'Lp', '2017-05-19')
        frame_size = (81, 81)
    elif planet == 'HR_8799':
        planet_part = ('HR_8799', 'Lp', '2012-08-25')
        frame_size = (101, 101)
    elif planet == 'PZ_Telescopii':
        planet_part = ('PZ_Telescopii', 'Lp', '2010-09-27')
        frame_size = (81, 81)
    else:
        raise ValueError(f'{planet} is not a valid planet name!')

    # Construct the full path to a data set
    file_name = f'stacked_{stacking_factor}.hdf'
    file_path = Path(data_dir, *planet_part, 'processed', file_name)

    # Load the data
    stack, parang, _ = load_data(file_path=file_path.as_posix(),
                                 frame_size=frame_size)

    return stack, parang
