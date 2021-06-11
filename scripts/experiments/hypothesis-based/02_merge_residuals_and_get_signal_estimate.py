"""
Merge partial residual FITS files and compute signal estimate.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import os
import time

from astropy.units import Quantity
from tqdm.auto import tqdm

import numpy as np

from hsr4hci.config import load_config
from hsr4hci.data import load_parang, load_metadata
from hsr4hci.derotating import derotate_combine
from hsr4hci.fits import read_fits, save_fits
from hsr4hci.masking import get_roi_mask, get_partial_roi_mask
from hsr4hci.merging import get_list_of_fits_file_paths
from hsr4hci.units import InstrumentUnitsContext


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nMERGE RESIDUAL FILES AND COMPUTE SIGNAL ESTIMATE\n', flush=True)

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

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    config = load_config(experiment_dir / 'config.json')
    print('Done!', flush=True)

    # Load metadata and parallactic angles
    metadata = load_metadata(**config['dataset'])
    parang = load_parang(**config['dataset'])

    # Define the unit conversion context for this data set
    instrument_unit_context = InstrumentUnitsContext(
        pixscale=Quantity(metadata['PIXSCALE'], 'arcsec / pixel'),
        lambda_over_d=Quantity(metadata['LAMBDA_OVER_D'], 'arcsec'),
    )

    # -------------------------------------------------------------------------
    # Collect and merge partial residual files
    # -------------------------------------------------------------------------

    # Define shortcuts
    n_frames = len(parang)
    x_size = config['dataset']['frame_size'][0]
    y_size = config['dataset']['frame_size'][1]
    frame_size = (x_size, y_size)

    # Construct the mask for the region of interest (ROI)
    with instrument_unit_context:
        roi_mask = get_roi_mask(
            mask_size=frame_size,
            inner_radius=Quantity(*config['roi_mask']['inner_radius']),
            outer_radius=Quantity(*config['roi_mask']['outer_radius']),
        )

    # Define directory that contains partial residuals
    partial_dir = experiment_dir / 'fits' / 'partial'

    # Collect paths to FITS files that we need to merge
    fits_file_paths = get_list_of_fits_file_paths(
        fits_dir=partial_dir, prefix='residuals'
    )

    # Initialize an array for the full residual stack
    residual_stack = np.full((n_frames, x_size, y_size), np.nan)

    # Loop over the FITS files that we need to merge
    print('\nMerging partial residuals:', flush=True)
    for file_path in tqdm(fits_file_paths, ncols=80):

        # Parse the file name to get the roi_split and n_roi_splits
        name = file_path.name.split('.')[0]
        roi_split = int(name.split('_')[1].split('-')[0]) - 1
        n_roi_splits = int(name.split('_')[1].split('-')[1])

        # Construct partial ROI mask that matches this file
        partial_roi_mask = get_partial_roi_mask(
            roi_mask=roi_mask,
            roi_split=roi_split,
            n_roi_splits=n_roi_splits,
        )

        # Read in the array with the partial residuals and place it at the
        # positions given by the partial ROI mask
        array = read_fits(file_path, return_header=False)
        residual_stack[:, partial_roi_mask] = array

    # Finally, save the merged FITS files
    print('\nSaving merged residuals to FITS...', end=' ', flush=True)
    file_path = experiment_dir / 'fits' / 'residuals.fits'
    save_fits(array=residual_stack, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Derotate frames to compute the signal estimate
    # -------------------------------------------------------------------------

    # Ensure that the results directory exists
    results_dir = experiment_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    # Compute signal estimate
    print('Computing signal estimate...', end=' ', flush=True)
    signal_estimate = derotate_combine(
        stack=residual_stack, parang=parang, mask=~roi_mask
    )
    print('Done!', flush=True)

    print('Saving signal estimate to FITS...', end=' ', flush=True)
    file_path = results_dir / 'signal_estimate.fits'
    save_fits(array=signal_estimate, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
