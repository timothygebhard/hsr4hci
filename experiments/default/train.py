"""
Train an HSR model, make predictions and derotate frames.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import h5py
import numpy as np
import os
import time

from hsr4hci.utils.adi_tools import derotate_frames
from hsr4hci.utils.config import load_config
from hsr4hci.utils.data import load_data, TrainTestSplitter
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
    print('\nTRAIN PIXEL-WISE MODELS\n', flush=True)

    # -------------------------------------------------------------------------
    # Load config and data
    # -------------------------------------------------------------------------

    # Load experiment config from JSON
    config = load_config('./config.json')

    # Load frames and parallactic angles from HDF file
    stack, parang = load_data(dataset_config=config['dataset'])

    # -------------------------------------------------------------------------
    # Train and apply models for all train / test splits
    # -------------------------------------------------------------------------

    # Set up a splitter for training / test
    train_test_splitter = \
        TrainTestSplitter(n_splits=config['dataset']['n_splits'],
                          split_type=config['dataset']['split_type'])

    # Instantiate hsr as None because we need it for the ROI mask (see below)
    hsr = None

    # Instantiate array to store the predictions
    predictions = np.full_like(stack, np.nan)

    # Loop over all train / test splits
    for train_idx, test_idx in train_test_splitter.split(stack):

        # Split the dataset
        train_stack, test_stack = stack[train_idx], stack[test_idx]
        train_parang, test_parang = parang[train_idx], parang[test_idx]

        # Instantiate and train model
        print(f'Training models (n_frames={len(train_stack)}):', flush=True)
        hsr = HalfSiblingRegression(config=config)
        hsr.train(training_stack=train_stack)
        print('', flush=True)

        # Apply model to test data and store predictions
        print(f'Making predictions (n_frames={len(test_stack)}):', flush=True)
        predictions[test_idx] = hsr.predict(test_stack=test_stack)
        print('', flush=True)

    # Compute the residuals
    print('Computing residuals...', end=' ', flush=True)
    residuals = stack - predictions
    print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Derotate both predictions and residuals by their parallactic angles
    # -------------------------------------------------------------------------

    # Get mask to apply after derotating
    # (This is why we had to initialize hsr it to None)
    roi_mask = hsr.m__roi_mask if hsr is not None else None

    print('Derotating predictions:', flush=True)
    derotated_predictions = derotate_frames(stack=predictions,
                                            parang=parang,
                                            verbose=True,
                                            mask=roi_mask)

    print('Derotating residuals:', flush=True)
    derotated_residuals = derotate_frames(stack=residuals,
                                          parang=parang,
                                          verbose=True,
                                          mask=roi_mask)

    # -------------------------------------------------------------------------
    # Save results to HDF and FITS
    # -------------------------------------------------------------------------

    # Ensure the results dir exists
    results_dir = os.path.join(config['experiment_dir'], 'results')
    Path(results_dir).mkdir(exist_ok=True)

    # Store predictions, residuals and parallactic angles in an HDF file
    print('\nSaving results to HDF...', end=' ', flush=True)
    hdf_file_path = os.path.join(results_dir, 'results.hdf')
    with h5py.File(hdf_file_path, 'w') as hdf_file:
        hdf_file.create_dataset(name='predictions',
                                data=predictions)
        hdf_file.create_dataset(name='derotated_predictions',
                                data=derotated_predictions)
        hdf_file.create_dataset(name='residuals',
                                data=residuals)
        hdf_file.create_dataset(name='derotated_residuals',
                                data=derotated_residuals)
        hdf_file.create_dataset(name='parang',
                                data=parang)
    print('Done!', flush=True)

    # Store predictions and residuals in a FITS file (both raw and derotated)
    print(f'Saving results to FITS...', end=' ', flush=True)
    for name, array in [('predictions', predictions),
                        ('residuals', residuals),
                        ('derotated_predictions', derotated_predictions),
                        ('derotated_residuals', derotated_residuals)]:
        fits_file_path = os.path.join(results_dir, f'{name}.fits')
        save_fits(array=array, file_path=fits_file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
