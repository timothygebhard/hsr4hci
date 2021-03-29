"""
Utility functions for loading data sets.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Dict, Optional, Tuple, Union

import h5py
import numpy as np

from hsr4hci.utils.config import get_datasets_dir
from hsr4hci.utils.general import crop_center, prestack_array
from hsr4hci.utils.observing_conditions import ObservingConditions


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

        # Select stack, parallactic angles and PSF template
        stack = np.array(hdf_file['stack']).astype(float)
        parang = np.array(hdf_file['parang']).astype(float)
        psf_template = np.array(hdf_file['psf_template']).astype(float)

        # Collect the observing conditions into a (temporary) dictionary
        _observing_conditions = dict()
        for key in hdf_file['observing_conditions']['interpolated'].keys():
            _observing_conditions[key] = np.array(
                hdf_file['observing_conditions']['interpolated'][key]
            ).astype(float)

        # Read the metadata into a dictionary
        metadata: Dict[str, Union[str, float]] = dict()
        for key in hdf_file['metadata'].keys():
            value = hdf_file['metadata'][key][()]
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            metadata[key] = value

    # -------------------------------------------------------------------------
    # Spatially crop the stack and apply temporal binning
    # -------------------------------------------------------------------------

    # Spatially crop the stack around the center to the desired frame size
    if frame_size is not None:
        stack = crop_center(stack, (-1, frame_size[0], frame_size[1]))

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
