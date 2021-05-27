"""
Utility functions for merging partial result files (FITS / HDF).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, List, Sequence
from warnings import warn, catch_warnings, filterwarnings

import os

from tqdm.auto import tqdm

import numpy as np

from hsr4hci.fits import read_fits
from hsr4hci.hdf import load_dict_from_hdf


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------


def get_list_of_fits_file_paths(fits_dir: Path, prefix: str) -> List[Path]:
    """
    Get a list of all FITS files in a given `fits_dir` whose file name
    begins with the given `prefix`.

    Args:
        fits_dir: Path to directory in which to look for FITS files.
        prefix: Only consider FITS files whose names begin with this.
            For example: "hypotheses" or "mean_mf".

    Returns:
        A list of Paths to the matching FITS files in `fits_dir`.
    """

    # Get a list of the paths to all FITS files in the given FITS directory
    # that start with the given prefix (e.g., "hypotheses" or "mean_mf")
    fits_file_names = filter(
        lambda _: _.endswith('.fits') and _.startswith(prefix),
        os.listdir(fits_dir),
    )
    fits_file_paths = [fits_dir / _ for _ in fits_file_names]

    # Perform a quick sanity check: Does the number of FITS files we found
    # match the number that we would expect based on the naming convention?
    # Reminder: The naming convention is "<prefix>_<split>-<n_splits>.fits".
    expected_number = int(fits_file_paths[0].name.split('-')[1].split('.')[0])
    actual_number = len(fits_file_paths)
    if expected_number != actual_number:
        warn(
            f'Naming convention suggests there should be {expected_number} '
            f'FITS files, but {actual_number} were found!'
        )

    return sorted(fits_file_paths)


def get_list_of_hdf_file_paths(
    hdf_dir: Path, prefix: str = 'results'
) -> List[Path]:
    """
    Get a list of all HDF files in a given `hdf_dir` whose file name
    begins with the given `prefix`.

    Args:
        hdf_dir: Path to directory in which to look for HDF files.
        prefix: Only consider HDF files whose names begin with this.
            Usually, we only need HDF files starting with "results".

    Returns:
        A list of Paths to the matching HDF files in `hdf_dir`.
    """

    # Get a list of the paths to all HDF files in the given HDF directory
    hdf_file_names = filter(
        lambda _: _.endswith('.hdf') and _.startswith(prefix),
        os.listdir(hdf_dir),
    )
    hdf_file_paths = [hdf_dir / _ for _ in hdf_file_names]

    # Perform a quick sanity check: Does the number of HDF files we found
    # match the number that we would expect based on the naming convention?
    # Reminder: The naming convention is "results_<split>-<n_splits>.hdf".
    expected_number = int(hdf_file_paths[0].name.split('-')[1].split('.')[0])
    actual_number = len(hdf_file_paths)
    if expected_number != actual_number:
        warn(
            f'Naming convention suggests there should be {expected_number} '
            f'HDF files, but {actual_number} were found!'
        )

    return sorted(hdf_file_paths)


def merge_hdf_files(
    hdf_file_paths: Sequence[Path],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Take a list of HDF files and merge all of them into a single dict.

    This function is intended to merge the (partial) results files that
    are produced by `hsr4hci.training.train_all_models()`; see there for
    more details on the expected internal structure of the HDF files.

    Args:
        hdf_file_paths: A list of paths to the HDF files to be merged.

    Returns:
        A dictionary containing the "full" (i.e., merged) results from
        all HDF files.
    """

    # Instantiate the dictionary which will hold the final results
    results: Dict[str, Dict[str, np.ndarray]] = dict()

    # Loop over all HDF files that we need to merge
    for hdf_file_path in tqdm(sorted(hdf_file_paths), ncols=80):

        # Load the HDF file to be merged
        hdf_file = load_dict_from_hdf(file_path=hdf_file_path)

        # Get the expected dimensions of the stack
        stack_shape = (
            int(hdf_file['stack_shape'][0]),
            int(hdf_file['stack_shape'][1]),
            int(hdf_file['stack_shape'][2]),
        )

        # Loop over the actual results in the HDF file:
        # The `key` is going to be either "default", or "0", ... "N" (i.e.,
        # the different signal_times for which we have trained a model).
        # The `value` is going to be a dictionary containing a `mask` as
        # well as the `residuals` (i.e., the partial results).
        for key, value in hdf_file['results'].items():

            # If necessary, create a new sub-dictionary in the results dict
            if key not in results.keys():
                results[key] = {'residuals': np.full(stack_shape, np.nan)}

            # Define shortcuts
            mask = value['mask']
            residuals = value['residuals']

            # Store the (partial) residuals in the correct location
            results[key]['residuals'][:, mask] = residuals

    return results


def merge_fits_files(fits_file_paths: List[Path]) -> np.ndarray:
    """
    Take a list of FITS files and merge all of them into a single array.

    This function is intended to merge the partial result files that are
    obtained in parallel with `hsr4hci.hypotheses.get_all_hypotheses()`
    and `hsr4hci.match_fractions.get_all_match_fractions()`.

    Merging works by stacking the arrays from the FITS files along a new
    axis and then taking the nanmean() along this axis. This, of course,
    assumes that each pixel only takes on a non-NaN value in at most one
    of the FITS files.

    Args:
        fits_file_paths: List of FITS files to be merged.

    Returns:
        A numpy array containing the merged arrays from all FITS files.
    """

    # Read in all FITS files as numpy arrays
    arrays = []
    for file_path in fits_file_paths:
        array = read_fits(file_path, return_header=False)
        arrays.append(array)

    # Stack and merge them along the first axis
    with catch_warnings():
        filterwarnings('ignore', r'Mean of empty slice')
        array = np.nanmean(arrays, axis=0)

    return array
