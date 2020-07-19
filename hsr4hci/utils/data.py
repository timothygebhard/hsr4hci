"""
Utility functions for loading data.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import h5py
import numpy as np

from hsr4hci.utils.config import get_data_dir
from hsr4hci.utils.general import crop_center


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------


def load_data(
    file_path: str,
    frame_size: Optional[Tuple[int, int]] = None,
    presubtract: Optional[str] = None,
    subsample: int = 1,
    **_: Any,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray],
    Dict[str, Union[str, float]],
]:
    """
    Load a dataset from the HDF file at the given `file_path`.

    Note: This function only works with HDF files that follow the
        specific assumptions regarding the file structure that are
        built into the script in the `create_data_and_baseline`
        directory in the `scripts` folder.

    Args:
        file_path: A string containing the path to the HDF file
            containing the data set to be loaded.
        frame_size: A tuple (width, height) of integers specifying the
            spatial size (in pixels) to which the stack will be cropped
            around the center. Dimensions should be odd numbers.
            If `None` is given, the frames are not cropped (default).
        presubtract: If this parameter is set to "mean" or "median",
            the mean (or median) along the time axis is subtracted from
            the stack before returning it.
        subsample: An integer specifying the subsampling factor for the
            stack. If set to n, only every n-th frame is kept. By
            default, all frames are kept (i.e., subsample=1).
        **_: Additional keyword arguments that will be ignored.

    Returns:
        A tuple `(stack, parang, psf_template, observing_conditions,
        metadata)`, containing numpy arrays with the frames, the
        parallactic angles and the unsaturated PSF template, as well as
        dictionaries with the observing conditions and the metadata.
    """

    # Read in the dataset from the HDf file
    with h5py.File(file_path, 'r') as hdf_file:

        # Select stack and parallactic angles and subsample as desired
        stack = np.array(hdf_file['/stack'][::subsample, ...])
        parang = np.array(hdf_file['/parang'][::subsample, ...])

        # Select the unsaturated PSF template
        psf_template = np.array(hdf_file['/psf_template']).squeeze()

        # Select the observing conditions
        observing_conditions: Dict[str, np.ndarray] = dict()
        for key in hdf_file['/observing_conditions'].keys():
            observing_conditions[key] = np.array(
                hdf_file['/observing_conditions'][key]
            )

        # Select the metadata
        metadata: Dict[str, Union[str, float]] = dict()
        for key in hdf_file.attrs.keys():
            metadata[key] = hdf_file.attrs[key]

    # Spatially crop the stack around the center to the desired frame size
    if frame_size is not None:
        stack = crop_center(stack, (-1, frame_size[0], frame_size[1]))

    # If desired, pre-subtract mean or median from the stack
    if presubtract == 'median':
        stack -= np.nanmedian(stack, axis=0)
    elif presubtract == 'mean':
        stack -= np.nanmean(stack, axis=0)

    return stack, parang, psf_template, observing_conditions, metadata


def load_default_data(
    planet: str,
    stacking_factor: int = 50,
    data_dir: Optional[str] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray],
    Dict[str, Union[str, float]],
]:
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
    stack, parang, psf_template, observing_conditions, metadata = load_data(
        file_path=file_path.as_posix(), frame_size=frame_size
    )

    return stack, parang, psf_template, observing_conditions, metadata
