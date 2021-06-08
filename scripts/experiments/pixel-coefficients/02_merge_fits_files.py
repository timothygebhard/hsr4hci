"""
Merge partial FITS files for the coefficients.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import os
import time

from hsr4hci.fits import save_fits
from hsr4hci.merging import get_list_of_fits_file_paths, merge_fits_files


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nMERGE COEFFICIENT FITS FILES\n', flush=True)

    # -------------------------------------------------------------------------
    # Set up parser to get command line arguments; get experiment directory
    # -------------------------------------------------------------------------

    # Set up parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-dir',
        type=str,
        required=True,
        metavar='PATH',
        help='(Absolute) path to experiment directory.',
    )
    args = parser.parse_args()

    # Get experiment directory
    experiment_dir = Path(os.path.expanduser(args.experiment_dir))
    if not experiment_dir.exists():
        raise NotADirectoryError(f'{experiment_dir} does not exist!')

    # -------------------------------------------------------------------------
    # Collect and merge FITS files with coefficients
    # -------------------------------------------------------------------------

    # Define directories for hypotheses
    fits_dir = experiment_dir / 'fits'
    partial_dir = fits_dir / 'partial'

    # Collect, merge and save coefficients
    print('Collecting and merging coefficients...', end=' ', flush=True)
    fits_file_paths = get_list_of_fits_file_paths(
        fits_dir=partial_dir, prefix='coefficients'
    )
    coefficients = merge_fits_files(fits_file_paths=fits_file_paths)
    file_path = fits_dir / 'coefficients.fits'
    save_fits(array=coefficients, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
