"""
Utility functions for loading data.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from astropy.units import Quantity

import h5py
import numpy as np

from hsr4hci.utils.config import get_data_dir
from hsr4hci.utils.general import crop_center
from hsr4hci.utils.psf import get_artificial_psf


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def load_data(
    target_name: str,
    filter_name: str,
    date: str,
    stacking_factor: int,
    frame_size: Optional[Tuple[int, int]] = None,
    presubtract: Optional[str] = None,
    subsample: int = 1,
    add_artificial_psf_template: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray],
    Dict[str, Union[str, float]],
]:
    """
    Load a data set specified by the target and filter name, date and
    stacking factor, and optionally apply some pre-processing.

    Note: This function only works with HDF files that follow the
        specific assumptions regarding the file structure that are
        built into the script in the `create_data_and_baseline`
        directory in the `scripts` folder.

    Args:
        target_name: Name of the target of the observation, that is, the
            name of the host star; e.g., "Beta_Pictoris".
        filter_name: Name of the filter that was used, e.g., "Lp".
        date: The date of the observation in format "YYYY-MM-DD".
        stacking_factor: The number of (raw) frames that were merged
            during the pre-processing of the data set.
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
        add_artificial_psf_template: Add an artificial PSF template if
            no real PSF  template is found. Default is True.

    Returns:
        A 5-tuple of the following form:
            `(stack, parang, psf_template,  obs_con, metadata)`,
        containing numpy arrays with the frames, the parallactic angles
        and the unsaturated PSF template, as well as *dictionaries* with
        the observing conditions and the metadata.
    """

    # Construct path to HDF file containing the data
    file_path = Path(
        get_data_dir(),
        target_name,
        filter_name,
        date,
        'processed',
        f'stacked__{stacking_factor}.hdf',
    )

    # Read in the dataset from the HDf file
    with h5py.File(file_path, 'r') as hdf_file:

        # Select stack and parallactic angles and subsample as desired
        stack = np.array(hdf_file['stack'][::subsample, ...])
        parang = np.array(hdf_file['parang'][::subsample, ...])

        # Select the unsaturated PSF template and ensure it is 2D
        psf_template = np.array(hdf_file['psf_template']).squeeze()
        if psf_template.ndim == 3:
            psf_template = np.mean(psf_template, axis=0)

        # Select the observing conditions
        observing_conditions: Dict[str, np.ndarray] = dict()
        for key in hdf_file['observing_conditions'].keys():
            observing_conditions[key] = np.array(
                hdf_file['observing_conditions'][key]
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

    # If necessary, create an artificial PSF
    if psf_template.shape[0] == 0 and add_artificial_psf_template:
        psf_template = get_artificial_psf(
            pixscale=Quantity(metadata['PIXSCALE'], 'arcsec / pixel'),
            lambda_over_d=Quantity(metadata['LAMBDA_OVER_D'], 'arcsec'),
        )

    return stack, parang, psf_template, observing_conditions, metadata


def load_default_data(
    planet: str, stacking_factor: int = 50
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
        planet: The name of the planet, given as "Planet_Name__Filter".
        stacking_factor: The pre-stacking factor of the data set. In
            general, this must be in (1, 5, 10, 25, 50, 100); the exact
            value depends on the data set.

    Returns:
        numpy arrays containing both the stack and the parallactic
        angles of the the requested data set.
    """

    # Hard-code some information about our most common data sets
    if planet == '51_Eridani__K1':
        target_name = '51_Eridani'
        filter_name = 'K1'
        date = '2015-09-25'
    elif planet == 'Beta_Pictoris__Lp':
        target_name = 'Beta_Pictoris'
        filter_name = 'Lp'
        date = '2013-02-01'
    elif planet == 'Beta_Pictoris__Mp':
        target_name = 'Beta_Pictoris'
        filter_name = 'Mp'
        date = '2012-11-26'
    elif planet == 'HIP_65426__Lp':
        target_name = 'HIP_65426'
        filter_name = 'Lp'
        date = '2017-05-19'
    elif planet == 'HR_8799__J':
        target_name = 'HR_8799'
        filter_name = 'J'
        date = '2014-08-14'
    elif planet == 'HR_8799__Lp':
        target_name = 'HR_8799'
        filter_name = 'Lp'
        date = '2011-09-01'
    elif planet == 'PZ_Telescopii__Lp':
        target_name = 'PZ_Telescopii'
        filter_name = 'Lp'
        date = '2010-09-27'
    elif planet == 'R_CrA__Lp':
        target_name = 'R_CrA'
        filter_name = 'Lp'
        date = '2018-06-07'
    else:
        raise ValueError(f'{planet} is not a valid planet name!')

    # Load the data and return them
    return load_data(
        target_name=target_name,
        filter_name=filter_name,
        date=date,
        stacking_factor=stacking_factor,
    )


def load_parang(
    target_name: str,
    filter_name: str,
    date: str,
    stacking_factor: int,
    **_: Any,
) -> np.ndarray:
    """
    Load the parallactic angles from the data set specified by the
    target and filter name, date and stacking factor.

    This is essentially just a subset of the `load_data()` function for
    cases where we do not want to load the entire stack into memory to
    get the parallactic angles.

    Args:
        target_name: Name of the target of the observation, that is, the
            name of the host star; e.g., "Beta_Pictoris".
        filter_name: Name of the filter that was used, e.g., "Lp".
        date: The date of the observation in format "YYYY-MM-DD".
        stacking_factor: The number of (raw) frames that were merged
            during the pre-processing of the data set.

    Returns:
        A numpy array containing the parallactic angles.
    """

    # Construct path to HDF file containing the data
    file_path = Path(
        get_data_dir(),
        target_name,
        filter_name,
        date,
        'processed',
        f'stacked__{stacking_factor}.hdf',
    )

    # Read in the dataset from the HDf file
    with h5py.File(file_path, 'r') as hdf_file:

        # Select stack and parallactic angles and subsample as desired
        parang = np.array(hdf_file['parang'])

    return parang
