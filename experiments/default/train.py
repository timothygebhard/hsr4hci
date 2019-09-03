"""
Train an HSR model (by default with forward modeling).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import time

from hsr4hci.utils.config import load_config
from hsr4hci.utils.data import load_data
from hsr4hci.utils.fits import save_fits
from hsr4hci.models.hsr import HalfSiblingRegression

from pathlib import Path


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

    # Load frames and parallactic angles from HDF file
    stack, parang, psf_template = load_data(dataset_config=config['dataset'])

    print(f'Training model using a stack of size {stack.shape}!', flush=True)

    # -------------------------------------------------------------------------
    # Set up and train model
    # -------------------------------------------------------------------------

    # Instantiate model
    hsr = HalfSiblingRegression(config=config)

    # Pre-compute PCA of pixels used for regression
    print('\nPre-computing PCA of predictor pixels:', flush=True)
    hsr.precompute_pca(stack=stack)
    
    # Train the model for all pixels in the ROI
    print('\nTraining models for all pixels in ROI:', flush=True)
    hsr.train(stack=stack, parang=parang, psf_template=psf_template)

    # -------------------------------------------------------------------------
    # Get the detection map and save it
    # -------------------------------------------------------------------------

    # Wrap this part in a try/except block for now so we don't lose our
    # training results simply because of an error in the detection map
    # calculation. Ultimately, this should no longer be necessary.
    try:

        # Get the detection map
        print(f'\nComputing detection map...', end=' ', flush=True)
        detection_map = hsr.get_detection_map()
        print('Done!', flush=True)

        # Ensure the results dir exists
        results_dir = os.path.join(config['experiment_dir'], 'results')
        Path(results_dir).mkdir(exist_ok=True)

        # Store the detection map to a FITS file
        print(f'Saving detection map to FITS...', end=' ', flush=True)
        fits_file_path = os.path.join(results_dir, f'detection_map.fits')
        save_fits(array=detection_map, file_path=fits_file_path)
        print('Done!', flush=True)

    except Exception as error_message:

        print('Something went wrong while computing the detection map!\n'
              'Here is the error message we got:\n\n')
        print(error_message, '\n\n')

    # -------------------------------------------------------------------------
    # Save the complete HSR model (this may take a while)
    # -------------------------------------------------------------------------

    print('\nSaving HSR model to pickle files:')
    hsr.save()

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
