"""
Merge individual HDF results files (from parallel training) into
a single HDF file.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import os
import time

from hsr4hci.hdf import save_dict_to_hdf
from hsr4hci.merging import get_list_of_hdf_file_paths, merge_hdf_files


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
    # Set up parser to get command line arguments
    # -------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-dir',
        type=str,
        required=True,
        metavar='PATH',
        help='(Absolute) path to experiment directory.',
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Define various variables
    # -------------------------------------------------------------------------

    # Get experiment directory
    experiment_dir = Path(os.path.expanduser(args.experiment_dir))
    if not experiment_dir.exists():
        raise NotADirectoryError(f'{experiment_dir} does not exist!')

    # Get paths to results and HDF directory
    hdf_dir = experiment_dir / 'hdf'
    partial_dir = hdf_dir / 'partial'

    # -------------------------------------------------------------------------
    # Collect file list, merge HDF files, and save the result
    # -------------------------------------------------------------------------

    # Get a list of all HDF files in the HDF directory
    print('Collecting HDF files to be merged...', end=' ', flush=True)
    hdf_file_paths = get_list_of_hdf_file_paths(hdf_dir=partial_dir)
    print('Done!', flush=True)

    # Merge the partial HDF files to get residuals
    print('Merging HDF files:', end=' ', flush=True)
    residuals = merge_hdf_files(hdf_file_paths=hdf_file_paths)
    print('', flush=True)

    # Save the final result
    print('\nSaving merged HDF file...', end=' ', flush=True)
    file_path = hdf_dir / 'residuals.hdf'
    save_dict_to_hdf(dictionary=residuals, file_path=file_path, mode='w')
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
