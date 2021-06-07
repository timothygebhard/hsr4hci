"""
Train default HSR model and store the pixel coefficients to FITS.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from itertools import product
from pathlib import Path

import argparse

from tqdm.auto import tqdm

import numpy as np
import time

from hsr4hci.base_models import BaseModelCreator
from hsr4hci.data import load_dataset
from hsr4hci.fits import save_fits
from hsr4hci.training import train_model_for_position


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nTRAIN HSR MODELS AND STORE MODEL COEFFICIENTS TO FITS\n')

    # -------------------------------------------------------------------------
    # Set up parser and parse command line arguments
    # -------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Name of the data set; e.g., "beta_pictoris__lp".',
    )
    parser.add_argument(
        '--binning-factor',
        type=int,
        default=10,
        help='Temporal binning factor for the stack.',
    )
    args = parser.parse_args()

    # Define shortcuts
    dataset = args.dataset
    binning_factor = args.binning_factor

    # -------------------------------------------------------------------------
    # Train a (default-style, i.e., no signal masking or fitting) HSR model
    # -------------------------------------------------------------------------

    # Load data set and remove planets
    print('Loading data set...', end=' ', flush=True)
    stack, parang, psf_template, obs_con, metadata = load_dataset(
        name_or_path=dataset,
        binning_factor=binning_factor,
        frame_size=(51, 51),
        remove_planets=True,
    )
    n_frames, x_size, y_size = stack.shape
    print('Done!', flush=True)

    # Define the base model that will be trained for each pixel
    base_model_creator = BaseModelCreator(
        **{
            "module": "sklearn.linear_model",
            "class": "RidgeCV",
            "parameters": {"fit_intercept": True, "alphas": [1e0, 1e6, 51]},
        }
    )

    # Prepare a 4D array in which we will store the coefficients
    coefficients = np.full((x_size, y_size, x_size, y_size), np.nan)

    # Loop over all pixels, train a model, and store the coefficients
    print('\nTraining models:', flush=True)
    for x, y in tqdm(
        list(product(np.arange(x_size), np.arange(y_size))), ncols=80
    ):

        # Train the model using ALL admissible pixels as predictors
        _, debug_dict = train_model_for_position(
            stack=stack,
            parang=parang,
            obscon_array=obs_con.as_array(None),
            position=(x, y),
            train_mode='default',
            signal_time=None,
            selection_mask_config={
                "radius_position": [1000.0, "pixel"],
                "radius_opposite": [1000.0, "pixel"],
            },
            psf_template=psf_template,
            n_train_splits=3,
            base_model_creator=base_model_creator,
        )

        # Get the selection mask, that is, the mask of the pixels used
        selection_mask = debug_dict['selection_mask']

        # Get the values of the coefficients and normalize to maximum
        pixel_coefs = np.mean(debug_dict['pixel_coefs'], axis=0)
        pixel_coefs /= np.nanmax(pixel_coefs)

        # Store the coefficients
        coefficients[x, y, selection_mask] = pixel_coefs

    # Ensure that the directory for the FITS files exists
    fits_dir = Path('./fits')
    fits_dir.mkdir(exist_ok=True)

    # Save the coefficients to FITS
    print('\nSaving results to FITS...', end=' ', flush=True)
    file_path = fits_dir / f'{dataset}__{binning_factor}.fits'
    save_fits(array=coefficients, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
