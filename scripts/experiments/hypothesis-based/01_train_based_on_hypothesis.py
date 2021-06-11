"""
Run a hypothesis-based version of half-sibling regression.
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
from hsr4hci.forward_modeling import add_fake_planet
from hsr4hci.masking import (
    get_roi_mask,
    get_positions_from_mask,
    get_partial_roi_mask,
)
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
    print('\nTRAIN HYPOTHESIS-BASED HALF-SIBLING REGRESSION\n', flush=True)

    # -------------------------------------------------------------------------
    # Load experiment configuration and data; parse command line arguments
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

    # Get experiment directory
    experiment_dir = Path(os.path.expanduser(args.experiment_dir))
    if not experiment_dir.exists():
        raise NotADirectoryError(f'{experiment_dir} does not exist!')

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    config = load_config(experiment_dir / 'config.json')
    if 'hypothesis' not in config.keys():
        raise RuntimeError('Experiment configuration contains no hypothesis!')
    print('Done!', flush=True)

    # Load frames, parallactic angles, etc. from HDF file
    print('Loading data set...', end=' ', flush=True)
    stack, parang, psf_template, observing_conditions, metadata = load_dataset(
        **config['dataset']
    )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Define shortcuts; activate unit conversions
    # -------------------------------------------------------------------------

    # Quantities related to the size of the data set
    n_frames, x_size, y_size = stack.shape
    frame_size = (x_size, y_size)

    # Command line arguments
    roi_split = args.roi_split
    n_roi_splits = args.n_roi_splits

    # Other shortcuts
    selected_keys = config['observing_conditions']['selected_keys']
    selection_mask_config = config['selection_mask']
    n_train_splits = int(config['n_train_splits'])

    # Define the unit conversion context for this data set
    instrument_unit_context = InstrumentUnitsContext(
        pixscale=Quantity(metadata['PIXSCALE'], 'arcsec / pixel'),
        lambda_over_d=Quantity(metadata['LAMBDA_OVER_D'], 'arcsec'),
    )

    # -------------------------------------------------------------------------
    # STEP 1: Compute expected stack based on hypothesis
    # -------------------------------------------------------------------------

    print('Computing hypothesized stack...', end=' ', flush=True)

    # Make sure the PSF template is correctly normalized
    psf_template /= np.max(psf_template)

    # Initialize the hypothesized stack and mask of affected pixels
    hypothesized_stack = np.zeros_like(stack)
    hypotheses = np.zeros(frame_size)
    affected_mask = np.full(frame_size, False)

    with instrument_unit_context:

        # Loop over potentially multiple planet hypotheses and add them
        for name, parameters in config['hypothesis'].items():

            # Define hypothesized planet position
            hypothesized_position = (
                Quantity(*parameters['separation']),
                Quantity(*parameters['position_angle']),
            )

            # Compute the corresponding forward model and normalize it
            hypothesized_stack = np.array(
                add_fake_planet(
                    stack=hypothesized_stack,
                    parang=parang,
                    psf_template=psf_template,
                    polar_position=hypothesized_position,
                    magnitude=0,
                    extra_scaling=1,
                    dit_stack=1,
                    dit_psf_template=1,
                    return_planet_positions=False,
                    interpolation='bilinear',
                )
            )
            hypothesized_stack /= np.max(hypothesized_stack)

            # Update the binary mask that we use to find pixels that contain a
            # "reasonable" amount of planet signal (the threshold is somewhat
            # arbitrary, of course)
            affected_mask = np.logical_or(
                affected_mask, (np.max(hypothesized_stack, axis=0) > 0.2)
            )

            # Find the hypotheses for the current stack, that is, the time at
            # which the signal peaks in each pixel
            hypotheses = np.maximum(
                hypotheses, np.argmax(hypothesized_stack, axis=0)
            )

    # Only keep the hypothesis for pixels with enough planet signal
    hypotheses[~affected_mask] = np.nan
    print('Done!', flush=True)

    # Ensure that the results directories exists
    fits_dir = experiment_dir / 'fits'
    fits_dir.mkdir(exist_ok=True)
    partial_fits_dir = fits_dir / 'partial'
    partial_fits_dir.mkdir(exist_ok=True)

    # Save hypotheses to HDF
    print('Saving hypotheses...', end=' ', flush=True)
    file_path = fits_dir / 'hypotheses.fits'
    if not file_path.exists():
        save_fits(hypotheses, file_path=file_path, overwrite=False)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # STEP 2: Train HSR models
    # -------------------------------------------------------------------------

    # Set up a BaseModelCreator to create instances of our base model
    base_model_creator = BaseModelCreator(**config['base_model'])

    # Construct the mask for the region of interest (ROI)
    with instrument_unit_context:
        roi_mask = get_roi_mask(
            mask_size=frame_size,
            inner_radius=Quantity(*config['roi_mask']['inner_radius']),
            outer_radius=Quantity(*config['roi_mask']['outer_radius']),
        )

    # Create partial ROI mask
    partial_roi_mask = get_partial_roi_mask(
        roi_mask=roi_mask,
        roi_split=args.roi_split,
        n_roi_splits=args.n_roi_splits,
    )

    # Initialize full residuals
    full_residuals = np.full(stack.shape, np.nan)

    # Loop over ROI, train models, and compute residuals
    print('\nTraining HSR models:', flush=True)
    for (x, y) in tqdm(get_positions_from_mask(partial_roi_mask), ncols=80):

        # Define values for train_mode, signal_time and expected_signal
        if np.isnan(hypotheses[x, y]):
            mode = 'default'
            signal_time = None
            expected_signal = None
        else:
            mode = config['train_mode']
            signal_time = int(hypotheses[x, y])
            expected_signal = hypothesized_stack[:, x, y]

        # Train the model for this pixel
        with instrument_unit_context:
            residuals, model_params = train_model_for_position(
                stack=stack,
                parang=parang,
                obscon_array=observing_conditions.as_array(selected_keys),
                position=(x, y),
                train_mode=mode,
                signal_time=signal_time,
                selection_mask_config=selection_mask_config,
                psf_template=psf_template,
                n_train_splits=n_train_splits,
                base_model_creator=base_model_creator,
                expected_signal=expected_signal,
            )
        full_residuals[:, x, y] = residuals

    print('\nSaving (partial) residuals to FITS...', end=' ', flush=True)
    file_name = f'residuals_{roi_split + 1:04d}-{n_roi_splits:04d}.fits'
    file_path = partial_fits_dir / file_name
    save_fits(full_residuals[:, partial_roi_mask], file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
