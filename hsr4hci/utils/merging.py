"""
Utility functions for merging partial result files (HDF files).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import cast, Dict, List, Sequence, Union
from warnings import warn

import os

from tqdm.auto import tqdm

import numpy as np

from hsr4hci.utils.hdf import load_dict_from_hdf


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_hdf_file_paths(hdf_dir: Path) -> List[Path]:
    """
    Collect the file paths of the partial result files (HDF files).

    Args:
        hdf_dir: Path to the directory that contains the HDF files.

    Returns:
        A list of paths to the partial result files.
    """

    # Get a list of the paths to all HDF files in the given HDF directory
    hdf_file_names = filter(lambda _: _.endswith('.hdf'), os.listdir(hdf_dir))
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

    return hdf_file_paths


def merge_result_files(
    hdf_file_paths: Sequence[Path],
) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
    """
    The standard training pipeline for the HSR was written with the
    possibility in mind to easily parallelize training on a cluster.
    In this scenario, we have multiple nodes, and each nodes processes
    only a subset of the ROI. Each node then stores its result in an
    HDF file with a special format that was optimized to minimize the
    overall storage that is required.
    This function takes a list of the paths to these "partial" result
    files and constructs the "full" result by creating a residual stack
    of the proper size and inserting the results from each partial HDF
    file at the correct locations.

    Args:
        hdf_file_paths: A list of paths to the HDF files to be merged.

    Returns:
        A dictionary containing the "full" (i.e., merged) results.
    """

    # Instantiate the dictionary which will hold the final results
    results: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]] = dict()

    # Loop over all HDF files that we need to merge
    for hdf_file_path in tqdm(sorted(hdf_file_paths), ncols=80):

        # Load the HDF file to be merged
        hdf_file = load_dict_from_hdf(file_path=hdf_file_path)

        # Get the dimensions of stack-shaped quantities
        stack_shape = (
            int(hdf_file['stack_shape'][0]),
            int(hdf_file['stack_shape'][1]),
            int(hdf_file['stack_shape'][2]),
        )

        # Loop over the contents of the file
        for key, value in hdf_file.items():

            # Special case: The signal_times data set is the same for all
            # results files and we only need to store it once
            if key == 'signal_times' and key not in results.keys():
                results[key] = value

            # In all other cases, the value itself should be a dict containing
            # several keys which each map to a numpy array
            elif isinstance(value, dict):

                # If necessary (i.e., for the first HDF file), create a new
                # sub-dictionary for the current group
                if key not in results.keys():
                    results[key] = dict()

                # Select the mask that tells us at which spatial positions we
                # have to place the values (e.g., residual time series) from
                # this group in this file.
                # This mask is not file-global, because it might differ for
                # the default and the signal masking-results: while the default
                # exists for every pixel, the signal masking results are only
                # available for pixels where the expected temporal length of
                # the signal is below a certain threshold.
                mask = value['mask']

                # Loop over the second-level dict, which can either contain
                # only the residuals and predictions (for the baseline), or
                # additionally also the signal_time and the signal_mask.
                for name, item in value.items():

                    # We do not need the information about the masks in the
                    # final merged HDF file, so we can skip it here
                    if name == 'mask':
                        continue

                    # For the first HDF file that we are merging, we need to
                    # initialize the correct shape of the result arrays
                    if name not in cast(dict, results[key]).keys():
                        results[key][name] = np.full(stack_shape, np.nan)

                    # Now we can use the mask to store the contents of the
                    # current HDF file at the right position in the overall
                    # results
                    results[key][name][:, mask] = item

    return results
