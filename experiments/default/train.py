"""
Train an HSR model
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import h5py
import numpy as np
import os
import time

from hsr4hci.utils.config import load_config
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
    print('\nTRAIN PIXEL-WISE MODELS\n', flush=True)

    # -------------------------------------------------------------------------
    # Load config and data
    # -------------------------------------------------------------------------

    config = load_config('./config.json')

    with h5py.File(config['dataset']['file_path'], 'r') as hdf_file:
        train_stack = \
            np.array(hdf_file[config['dataset']['stack_key']][0:-1:2])
        test_stack = \
            np.array(hdf_file[config['dataset']['stack_key']][1:-1:2])
        train_parang = \
            np.array(hdf_file[config['dataset']['parang_key']][0:-1:2])
        test_parang = \
            np.array(hdf_file[config['dataset']['parang_key']][1:-1:2])

    # -------------------------------------------------------------------------
    # Train model
    # -------------------------------------------------------------------------

    print('Training model(s):', flush=True)
    hsr = HalfSiblingRegression(config=config)
    hsr.train(training_stack=train_stack)
    print('', flush=True)

    # -------------------------------------------------------------------------
    # Apply model to test data
    # -------------------------------------------------------------------------

    print('Making predictions:', flush=True)
    predictions = hsr.predict(test_stack=test_stack)
    print('', flush=True)

    # -------------------------------------------------------------------------
    # Save predictions and parallactic angles to HDF
    # -------------------------------------------------------------------------

    # Ensure the results dir exists
    results_dir = os.path.join(config['experiment_dir'], 'results')
    Path(results_dir).mkdir(exist_ok=True)

    print('Saving predictions to HDF file...', end=' ', flush=True)
    file_path = os.path.join(results_dir, 'predictions.hdf')
    with h5py.File(file_path, 'w') as hdf_file:
        hdf_file.create_dataset(name='predictions', data=predictions)
        hdf_file.create_dataset(name='parang', data=test_parang)
        hdf_file.create_dataset(name='test_stack', data=test_stack)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
