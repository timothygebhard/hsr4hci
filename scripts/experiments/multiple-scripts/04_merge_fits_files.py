"""
Merge partial FITS files for the hypotheses and the match fractions
into single FITS files.
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
    fits_file_paths = get_list_of_fits_file_paths(
        fits_dir=partial_dir, prefix='hypotheses'
    )
    hypotheses = merge_fits_files(fits_file_paths=fits_file_paths)
    file_path = hypotheses_dir / 'hypotheses.fits'
    save_fits(array=hypotheses, file_path=file_path)
    print('Done!', flush=True)

    # Collect, merge and save similarities
    print('Collecting and merging similarities...', end=' ', flush=True)
    fits_file_paths = get_list_of_fits_file_paths(
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
    fits_file_paths = get_list_of_fits_file_paths(
        fits_dir=partial_dir, prefix='mean_mf'
    )
    mean_mf = merge_fits_files(fits_file_paths=fits_file_paths)
    file_path = match_fractions_dir / 'mean_mf.fits'
    save_fits(array=mean_mf, file_path=file_path)
    print('Done!', flush=True)

    # Collect, merge and save similarities
    print('Collecting and merging median MFs...', end=' ', flush=True)
    fits_file_paths = get_list_of_fits_file_paths(
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
