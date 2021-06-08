"""
Train HSR models and store pixel coefficients to FITS.
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

from hsr4hci.base_models import BaseModelCreator
from hsr4hci.config import load_config
from hsr4hci.data import load_dataset
from hsr4hci.fits import save_fits
from hsr4hci.masking import get_partial_roi_mask, get_positions_from_mask
from hsr4hci.training import train_model_for_position
from hsr4hci.units import InstrumentUnitsContext


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nTRAIN HALF-SIBLING REGRESSION MODELS\n', flush=True)

    # -------------------------------------------------------------------------
    # Set up parser to get command line arguments
    # -------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-dir',
        type=str,
        required=True,
        metavar='PATH',
        help='(Absolute) path to experiment directory.',
    )
    parser.add_argument(
        '--roi-split',
        type=int,
        default=0,
        help='Index of the split to process; must be in [0, n_roi_splits).',
    )
    parser.add_argument(
        '--n-roi-splits',
        type=int,
        default=1,
        help=(
            'Number of splits into which the ROI is divided (e.g., for '
            'parallel training).'
        ),
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load experiment configuration and data
    # -------------------------------------------------------------------------

    # Get experiment directory
    experiment_dir = Path(os.path.expanduser(args.experiment_dir))
    if not experiment_dir.exists():
        raise NotADirectoryError(f'{experiment_dir} does not exist!')

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    config = load_config(experiment_dir / 'config.json')
    print('Done!', flush=True)

    # Load frames, parallactic angles, etc. from HDF file
    print('Loading data set...', end=' ', flush=True)
    stack, parang, psf_template, observing_conditions, metadata = load_dataset(
        **config['dataset']
    )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Define various useful shortcuts; activate unit conversions
    # -------------------------------------------------------------------------

    # Quantities related to the size of the data set
    n_frames, x_size, y_size = stack.shape
    frame_size = (x_size, y_size)

    # Metadata of the data set
    pixscale = float(metadata['PIXSCALE'])
    lambda_over_d = float(metadata['LAMBDA_OVER_D'])

    # Other shortcuts
    selected_keys = config['observing_conditions']['selected_keys']
    roi_split = args.roi_split
    n_roi_splits = args.n_roi_splits

    # Define the unit conversion context for this data set
    instrument_units_context = InstrumentUnitsContext(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
    )

    # -------------------------------------------------------------------------
    # Train half-sibling regression models
    # -------------------------------------------------------------------------

    # Set up a BaseModelCreator to create instances of our base model
    base_model_creator = BaseModelCreator(**config['base_model'])

    # Prepare a 4D array in which we will store the pixel coefficients
    coefficients = np.full((x_size, y_size, x_size, y_size), np.nan)

    # Define the the subset of the frame which we will process
    partial_roi_mask = get_partial_roi_mask(
        roi_mask = np.full((x_size, y_size), True),
        roi_split=roi_split,
        n_roi_splits=n_roi_splits
    )

    # Train models and get residuals for them
    print('\nTraining models:', flush=True)
    for x, y in tqdm(get_positions_from_mask(partial_roi_mask), ncols=80):

        # Train the model using ALL admissible pixels as predictors
        with instrument_units_context:
            _, debug_dict = train_model_for_position(
                stack=stack,
                parang=parang,
                obscon_array=observing_conditions.as_array(selected_keys),
                position=(x, y),
                train_mode='default',
                signal_time=None,
                selection_mask_config={
                    "radius_position": [1000.0, "pixel"],
                    "radius_opposite": [1000.0, "pixel"],
                },
                psf_template=psf_template,
                n_train_splits=config['n_train_splits'],
                base_model_creator=base_model_creator,
            )

        # Get the selection mask, that is, the mask of the pixels used
        selection_mask = debug_dict['selection_mask']

        # Get the values of the coefficients and normalize to maximum
        pixel_coefs = np.mean(debug_dict['pixel_coefs'], axis=0)
        pixel_coefs /= np.nanmax(np.abs(pixel_coefs))

        # Store the coefficients
        coefficients[x, y, selection_mask] = pixel_coefs

    print()

    # -------------------------------------------------------------------------
    # Save residuals to an HDF file
    # -------------------------------------------------------------------------

    # Prepare results directory
    results_dir = experiment_dir / 'fits' / 'partial'
    results_dir.mkdir(exist_ok=True, parents=True)

    # Sav the (partial) pixel coefficients as a FITS file
    print('Saving (partial) pixel coefficients...', end=' ', flush=True)
    file_name = f'coefficients_{roi_split + 1:04d}-{n_roi_splits:04d}.fits'
    file_path = results_dir / file_name
    save_fits(array=coefficients, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
