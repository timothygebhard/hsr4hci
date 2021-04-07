"""
Utility functions for loading data sets.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Dict, Optional, Tuple, Union

import h5py
import numpy as np

from hsr4hci.config import get_datasets_dir
from hsr4hci.general import prestack_array
from hsr4hci.observing_conditions import ObservingConditions


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def load_dataset(
    name: str,
    binning_factor: int = 1,
    frame_size: Optional[Tuple[int, int]] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    ObservingConditions,
    Dict[str, Union[str, float]],
]:
    """
    Load the data set with the given `name`, optionally cropping and
    temporally binning the data.

    Args:
        name: Name of the data set (e.g., "beta_pictoris__lp").
        binning_factor: Number of frames that should be temporally
            binned ("pre-stacked") using a block-wise mean.
        frame_size: A tuple (width, height) of integers specifying the
            spatial size (in pixels) to which the stack will be cropped
            around the center. Dimensions should be odd numbers.
            If `None` is given, the frames are not cropped (default).

    Returns:
        A 5-tuple of the following form:
            `(stack, parang, psf_template, obs_con, metadata)`,
        containing numpy arrays with the frames, the parallactic angles
        and the unsaturated PSF template, as well as the observing
        conditions as a `ObservingConditions` object and the metadata
        as a dictionary.
    """

    # -------------------------------------------------------------------------
    # Read in data set from HDF file
    # -------------------------------------------------------------------------

    # Construct path to HDF file containing the data set
    file_path = get_datasets_dir() / name / 'output' / f'{name}.hdf'

    # Read in the data set from the HDF file
    with h5py.File(file_path, 'r') as hdf_file:

        # Get shape of the stack in the HDF file
        stack_shape = hdf_file['stack'].shape

        # Define target shape (= stack shape after spatial cropping)
        if frame_size is not None:
            target_shape = (-1, frame_size[0], frame_size[1])
        else:
            target_shape = (-1, -1, -1)

        # Compute slices that can be used to select a cropped version of the
        # stack directly (this is much faster than loading the entire stack
        # into memory to crop it there). The code here is basically the same
        # as the one in `hsr4hci.general.crop_center()`.
        slices = list()
        for old_len, new_len in zip(stack_shape, target_shape):
            if new_len > old_len:
                start = None
                end = None
            else:
                start = old_len // 2 - new_len // 2 if new_len != -1 else None
                end = start + new_len if start is not None else None
            slices.append(slice(start, end))

        # Select stack, parallactic angles and PSF template
        stack = np.array(hdf_file['stack'][tuple(slices)])
        parang = np.array(hdf_file['parang'])
        psf_template = np.array(hdf_file['psf_template']).squeeze()

        # Ensure that the PSF template is two-dimensional now; otherwise this
        # can result in weird errors that are hard to debug
        if psf_template.ndim != 2:
            raise RuntimeError(
                f'psf_template is not 2D! (shape = {psf_template.shape}'
            )

        # Collect the observing conditions into a (temporary) dictionary
        _observing_conditions = dict()
        for key in hdf_file['observing_conditions']['interpolated'].keys():
            _observing_conditions[key] = np.array(
                hdf_file['observing_conditions']['interpolated'][key]
            )

        # Read the metadata into a dictionary
        metadata: Dict[str, Union[str, float]] = dict()
        for key in hdf_file['metadata'].keys():
            value = hdf_file['metadata'][key][()]
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            metadata[key] = value

    # -------------------------------------------------------------------------
    # Apply temporal binning to the stack
    # -------------------------------------------------------------------------

    # Apply temporal binning to stack, parang and observing conditions
    stack = prestack_array(array=stack, stacking_factor=binning_factor)
    parang = prestack_array(array=parang, stacking_factor=binning_factor)
    for key in _observing_conditions.keys():
        _observing_conditions[key] = prestack_array(
            array=_observing_conditions[key], stacking_factor=binning_factor
        )

    # Convert the observing conditions into an ObservingConditions object
    observing_conditions = ObservingConditions(_observing_conditions)

    return stack, parang, psf_template, observing_conditions, metadata


def load_parang(name: str, binning_factor: int = 1, **_: Any) -> np.ndarray:
    """
    Load (only) the parallactic angles for the given data set.

    Args:
        name: Name of the data set (e.g., "beta_pictoris__lp").
        binning_factor: Number of frames that should be temporally
            binned ("pre-stacked") using a block-wise mean.

    Returns:
        A numpy array containing the parallactic angles.
    """

    # Read in the data set from the HDF file
    file_path = get_datasets_dir() / name / 'output' / f'{name}.hdf'
    with h5py.File(file_path, 'r') as hdf_file:
        parang = np.array(hdf_file['parang']).astype(float)

    # Temporally bin the parallactic angles
    parang = prestack_array(array=parang, stacking_factor=binning_factor)

    return parang


def load_psf_template(name: str, **_: Any) -> np.ndarray:
    """
    Load (only) the unsaturated PSF template for the given data set.

    Args:
        name: Name of the data set (e.g., "beta_pictoris__lp").

    Returns:
        A numpy array containing the unsaturated PSF template.
    """

    # Read in the data set from the HDF file
    file_path = get_datasets_dir() / name / 'output' / f'{name}.hdf'
    with h5py.File(file_path, 'r') as hdf_file:
        psf_template = np.array(hdf_file['psf_template']).astype(float)

    return psf_template


def load_metadata(name: str, **_: Any) -> dict:
    """
    Load (only) the metadata for the given data set.

    Args:
        name: Name of the data set (e.g., "beta_pictoris__lp").

    Returns:
        A dictionary containing the metadata.
    """

    # Initialize metadata
    metadata: Dict[str, Union[str, float]] = dict()

    # Read in the data set from the HDF file
    file_path = get_datasets_dir() / name / 'output' / f'{name}.hdf'
    with h5py.File(file_path, 'r') as hdf_file:
        for key in hdf_file['metadata'].keys():
            value = hdf_file['metadata'][key][()]
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            metadata[key] = value

    return metadata
