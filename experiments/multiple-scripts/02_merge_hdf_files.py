"""
Merge individual HDF results files (from parallel training) into
a single HDF file.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import os
import time

from hsr4hci.hdf import save_dict_to_hdf
from hsr4hci.merging import get_hdf_file_paths, merge_result_files


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
    # Collect file list, merge HDF files, and save the result
    # -------------------------------------------------------------------------

    # Get a list of all HDF files in the HDF directory
    print('Collecting HDF files to be merged...', end=' ', flush=True)
    hdf_file_paths = get_hdf_file_paths(hdf_dir=hdf_dir)
    print('Done!', flush=True)

    # Merge the partial HDF files
    print('Merging HDF files:', end=' ', flush=True)
    results = merge_result_files(hdf_file_paths=hdf_file_paths)
    print('', flush=True)

    # Save the final result
    print('\nSaving merged HDF file...', end=' ', flush=True)
    file_path = results_dir / 'results.hdf'
    save_dict_to_hdf(dictionary=results, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
