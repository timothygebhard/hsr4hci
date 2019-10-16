"""
Train an HSR model (by default with forward modeling).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import os
import time

from hsr4hci.utils.config import load_config
from hsr4hci.utils.data import load_data
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
    print(f'pca_mode:  \t {config["experiment"]["sources"]["pca_mode"]}')

    # -------------------------------------------------------------------------
    # Set up and train model
    # -------------------------------------------------------------------------

    print('\n\nTraining model:')
    print(80 * '-')

    # Instantiate model with this config
    hsr = HalfSiblingRegression(config=config)

    # Train the model for all pixels in the ROI
    print('\nTraining models for all pixels in ROI:', flush=True)
    hsr.train(stack=stack, parang=parang, psf_template=psf_template)

    print(80 * '-')
    print('\n')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
