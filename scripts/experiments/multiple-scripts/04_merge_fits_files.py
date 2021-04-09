"""
Merge partial FITS files for the hypotheses and the match fractions
into single FITS files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import List

import argparse
import os
import time
import warnings

import numpy as np

from hsr4hci.fits import read_fits, save_fits


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_fits_file_list(fits_dir: Path, prefix: str) -> List[Path]:

    # Get a list of the paths to all FITS files in the given FITS directory
    # that start with the given prefix (e.g., "hypotheses" or "mean_mf")
    fits_file_names = filter(
        lambda _: _.endswith('.fits') and _.startswith(prefix),
        os.listdir(fits_dir)
    )
    fits_file_paths = [fits_dir / _ for _ in fits_file_names]

    # Perform a quick sanity check: Does the number of FITS files we found
    # match the number that we would expect based on the naming convention?
    # Reminder: The naming convention is "<prefix>_<split>-<n_splits>.fits".
    expected_number = int(fits_file_paths[0].name.split('-')[1].split('.')[0])
    actual_number = len(fits_file_paths)
    if expected_number != actual_number:
        warnings.warn(
            f'Naming convention suggests there should be {expected_number} '
            f'FITS files, but {actual_number} were found!'
        )

    return fits_file_paths


def merge_fits_files(fits_file_paths: List[Path]) -> np.ndarray:

    # Read in all FITS files as numpy arrays
    arrays = []
    for file_path in fits_file_paths:
        array = np.asarray(read_fits(file_path))
        arrays.append(array)

    # Stack and merge them along the first axis
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        array = np.nanmean(arrays)

    return array


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nMERGE FITS FILES\n', flush=True)

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
    # Collect and merge hypotheses files
    # -------------------------------------------------------------------------

    # Define directories for hypotheses
    hypotheses_dir = experiment_dir / 'hypotheses'
    partial_dir = hypotheses_dir / 'partial'

    # Collect, merge and save hypotheses
    print('Collecting and merging hypotheses...', end=' ', flush=True)
    fits_file_paths = get_fits_file_list(
        fits_dir=partial_dir, prefix='hypotheses'
    )
    hypotheses = merge_fits_files(fits_file_paths=fits_file_paths)
    file_path = hypotheses_dir / 'hypotheses.fits'
    save_fits(array=hypotheses, file_path=file_path)
    print('Done!', flush=True)

    # Collect, merge and save similarities
    print('Collecting and merging similarities...', end=' ', flush=True)
    fits_file_paths = get_fits_file_list(
        fits_dir=partial_dir, prefix='similarities'
    )
    similarities = merge_fits_files(fits_file_paths=fits_file_paths)
    file_path = hypotheses_dir / 'similarities.fits'
    save_fits(array=similarities, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Collect and merge match fraction files
    # -------------------------------------------------------------------------

    # Define directories for match fractions
    match_fractions_dir = experiment_dir / 'match_fractions'
    partial_dir = match_fractions_dir / 'partial'

    # Collect, merge and save hypotheses
    print('Collecting and merging mean MFs...', end=' ', flush=True)
    fits_file_paths = get_fits_file_list(
        fits_dir=partial_dir, prefix='mean_mf'
    )
    mean_mf = merge_fits_files(fits_file_paths=fits_file_paths)
    file_path = match_fractions_dir / 'mean_mf.fits'
    save_fits(array=mean_mf, file_path=file_path)
    print('Done!', flush=True)

    # Collect, merge and save similarities
    print('Collecting and merging median MFs...', end=' ', flush=True)
    fits_file_paths = get_fits_file_list(
        fits_dir=partial_dir, prefix='median_mf'
    )
    median_mf = merge_fits_files(fits_file_paths=fits_file_paths)
    file_path = match_fractions_dir / 'median_mf.fits'
    save_fits(array=median_mf, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
