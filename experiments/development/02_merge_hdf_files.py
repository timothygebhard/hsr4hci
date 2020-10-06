"""
Merge individual HDF results files (from parallel training) into
a single HDF file.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, Union

import os
import time

from tqdm.auto import tqdm

import numpy as np

from hsr4hci.utils.hdf import load_dict_from_hdf, save_dict_to_hdf


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nMERGE HDF RESULT FILES\n', flush=True)

    # -------------------------------------------------------------------------
    # Define various variables
    # -------------------------------------------------------------------------

    experiment_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    results_dir = experiment_dir / 'results'
    hdf_dir = results_dir / 'hdf'

    # -------------------------------------------------------------------------
    # Get a list of all files that we need to merge
    # -------------------------------------------------------------------------

    # Get a list of all HDF files in the HDF directory
    print('Collecting HDF files to be merged...', end=' ', flush=True)
    hdf_file_paths = [
        hdf_dir / hdf_file_name
        for hdf_file_name in filter(
            lambda _: _.endswith('hdf'), os.listdir(hdf_dir.as_posix())
        )
    ]
    print('Done!', flush=True)

    # Perform a quick sanity check: Does the number of HDF files we found
    # match the number that we would expect based on the naming convention?
    expected_number = int(hdf_file_paths[0].name.split('-')[1].split('.')[0])
    actual_number = len(hdf_file_paths)
    if expected_number != actual_number:
        print(f'\nWARNING: Naming convention suggest that there should be '
              f'{expected_number} HDF files but {actual_number} were found!')

    # Instantiate the dictionary which will hold the final results
    results: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]] = dict()

    # Loop over all HDF files that we need to merge
    print('\nProcessing individual HDF result files:', flush=True)
    for hdf_file_path in tqdm(sorted(hdf_file_paths), ncols=80):

        # Load the HDF to be merged
        hdf_file = load_dict_from_hdf(file_path=hdf_file_path.as_posix())

        # Get the dimensions of stack-shaped quantities
        stack_shape = (
            int(hdf_file['stack_shape'][0]),
            int(hdf_file['stack_shape'][1]),
            int(hdf_file['stack_shape'][2]),
        )
        frame_shape = stack_shape[1:]

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
                # the baseline and the signal masking-results: while the
                # baseline exists for every pixel, the signal masking results
                # are only available for pixels where the expected temporal
                # length of the signal is below a certain threshold.
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
                    if name not in results[key].keys():
                        if item.ndim == 1:
                            results[key][name] = np.full(frame_shape, np.nan)
                        else:
                            results[key][name] = np.full(stack_shape, np.nan)

                    # Now we can use the mask to store the contents of the
                    # current HDF file at the right position in the overall
                    # results
                    if item.ndim == 1:
                        results[key][name][mask] = item
                    else:
                        results[key][name][:, mask] = item

    # Save the final result
    print('\nSaving merged HDF file...', end=' ', flush=True)
    file_path = results_dir / 'results.hdf'
    save_dict_to_hdf(dictionary=results, file_path=file_path.as_posix())
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
