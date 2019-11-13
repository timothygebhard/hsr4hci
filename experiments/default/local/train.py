"""
Train an HSR model (by default with forward modeling).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import os
import time

import numpy as np

from hsr4hci.utils.config import load_config
from hsr4hci.utils.data import load_data
from hsr4hci.utils.fits import save_fits
from hsr4hci.models.hsr import HalfSiblingRegression


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nTRAIN HALF-SIBLING REGRESSION MODEL\n', flush=True)

    # -------------------------------------------------------------------------
    # Load config and data
    # -------------------------------------------------------------------------

    # Load experiment config from JSON
    config = load_config('./config.json')

    # Ensure the results dir exists
    results_dir = os.path.join(config['experiment_dir'], 'results')
    Path(results_dir).mkdir(exist_ok=True)

    # Load frames and parallactic angles from HDF file
    stack, parang, psf_template = load_data(dataset_config=config['dataset'])

    # Print some basic training information
    print(f'Stack size:\t {stack.shape}', flush=True)
    print(f'Model type:\t {config["experiment"]["model"]["module"]}.'
          f'{config["experiment"]["model"]["class"]}', flush=True)

    # -------------------------------------------------------------------------
    # Ensure the results dir exists
    # -------------------------------------------------------------------------

    results_dir = os.path.join(config['experiment_dir'], 'results')
    Path(results_dir).mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Set up and train model
    # -------------------------------------------------------------------------

    # Instantiate model with this config
    hsr = HalfSiblingRegression(config=config)

    # Train the model for all pixels in the ROI
    print('\nTraining models for all pixels in ROI:', flush=True)
    hsr.train(stack=stack, parang=parang, psf_template=psf_template)

    # -------------------------------------------------------------------------
    # Get the detection map and save it
    # -------------------------------------------------------------------------

    # Get detection map
    print(f'\nComputing detection map:', flush=True)
    detection_map = hsr.get_detection_map()

    # Store the detection map to a FITS file
    print(f'Saving detection map to FITS...', end=' ', flush=True)
    fits_file_path = os.path.join(results_dir, 'detection_map.fits')
    save_fits(array=detection_map, file_path=fits_file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Get the residual stack and save it
    # -------------------------------------------------------------------------

    # Get residual stack
    print(f'\nComputing residual stack:', flush=True)
    residual_stack = hsr.get_residual_stack(stack=stack)

    # Store the residual stack to a FITS file
    print(f'Saving residual stack to FITS...', end=' ', flush=True)
    fits_file_path = os.path.join(results_dir, 'residual_stack.fits')
    save_fits(array=residual_stack, file_path=fits_file_path)
    print('Done!', flush=True)

    # Compute average residual stack
    print('Averaging residual stack...', end=' ', flush=True)
    average_residuals = np.nanmedian(residual_stack, axis=0)
    print('Done!', flush=True)

    # Save average residual stack
    print('Saving average residuals...', end=' ', flush=True)
    fits_file_path = os.path.join(results_dir, 'average_residuals.fits')
    save_fits(average_residuals, fits_file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
