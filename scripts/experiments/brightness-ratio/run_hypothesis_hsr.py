"""
Inject fake planet and run a hypothesis-bases HSR pipeline.
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
from hsr4hci.derotating import derotate_combine
from hsr4hci.fits import save_fits
from hsr4hci.forward_modeling import add_fake_planet
from hsr4hci.masking import get_roi_mask, get_positions_from_mask
from hsr4hci.positions import get_injection_position
from hsr4hci.psf import get_psf_fwhm
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
    # Parse command line arguments
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

    # -------------------------------------------------------------------------
    # Load experiment configuration and data set
    # -------------------------------------------------------------------------

    # Get experiment directory
    experiment_dir = Path(os.path.expanduser(args.experiment_dir))
    if not experiment_dir.exists():
        raise NotADirectoryError(f'{experiment_dir} does not exist!')

    # Get path to results directory
    results_dir = experiment_dir / 'results'
    results_dir.mkdir(exist_ok=True)

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

    # Define the unit conversion context for this data set
    instrument_units_context = InstrumentUnitsContext(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
    )

    # Fit the FWHM of the PSF (in pixels)
    psf_fwhm = get_psf_fwhm(psf_template)

    # -------------------------------------------------------------------------
    # Inject a fake planet into the stack; construct "hypotheses"
    # -------------------------------------------------------------------------

    # Get injection parameters
    contrast = config['injection']['contrast']
    separation = config['injection']['separation']
    azimuthal_position = config['injection']['azimuthal_position']

    # If any parameter is None, skip the injection...
    if contrast is None or separation is None or azimuthal_position is None:
        signal_stack = np.full_like(stack, np.nan)
        hypotheses = np.full((stack.shape[1], stack.shape[2]), np.nan)
        print('Skipping injection of a fake planet!', flush=True)

    # ... otherwise, add a fake planet with given parameters to the stack
    else:

        # Convert separation from units of FWHM to pixel
        separation *= psf_fwhm

        # Compute position at which to inject the fake planet
        print('Computing injection position...', end=' ', flush=True)
        injection_position = get_injection_position(
            separation=Quantity(separation, 'pixel'),
            azimuthal_position=azimuthal_position,
        )
        print(
            f'Done! (separation = {separation:.1f} pixel, '
            f'azimuthal_position = {azimuthal_position})',
            flush=True,
        )

        # Inject the fake planet at the injection_position
        print('Injecting fake planet...', end=' ', flush=True)
        signal_stack = np.asarray(
            add_fake_planet(
                stack=np.zeros_like(stack),
                parang=parang,
                psf_template=psf_template,
                polar_position=injection_position,
                magnitude=contrast,
                extra_scaling=1,
                dit_stack=float(metadata['DIT_STACK']),
                dit_psf_template=float(metadata['DIT_PSF_TEMPLATE']),
                return_planet_positions=False,
                interpolation='bilinear',
            )
        )
        stack += signal_stack
        print('Done!', flush=True)

        # Find the hypotheses, that is, for every pixel find the time at
        # which the planet signal peaks
        hypotheses = np.asarray(np.argmax(signal_stack, axis=0)).astype(float)

        # Set all pixels without a significant planet signal to NaN to
        # indicate that we only need to train a default model
        mask = np.max(signal_stack / np.max(signal_stack), axis=0) < 0.2
        hypotheses[mask] = np.nan

    # -------------------------------------------------------------------------
    # STEP 1: Train HSR models
    # -------------------------------------------------------------------------

    # Set up a BaseModelCreator to create instances of our base model
    base_model_creator = BaseModelCreator(**config['base_model'])

    # Construct the mask for the region of interest (ROI)
    with instrument_units_context:
        roi_mask = get_roi_mask(
            mask_size=frame_size,
            inner_radius=Quantity(*config['roi_mask']['inner_radius']),
            outer_radius=Quantity(*config['roi_mask']['outer_radius']),
        )

    # Instantiate an array for the residuals (i.e., the training results)
    residuals = np.full_like(stack, np.nan)

    print('\nTraining models:', flush=True)
    for (x, y) in tqdm(get_positions_from_mask(roi_mask), ncols=80):

        # Define values for train_mode, signal_time and expected_signal
        if np.isnan(hypotheses[x, y]):
            mode = 'default'
            signal_time = None
            expected_signal = None
        else:
            mode = config['train_mode']
            signal_time = int(hypotheses[x, y])
            expected_signal = signal_stack[:, x, y] / np.max(signal_stack)

        # Train the model for this pixel and store the residuals
        with instrument_units_context:
            residuals_for_position, _ = train_model_for_position(
                stack=stack,
                parang=parang,
                obscon_array=observing_conditions.as_array(selected_keys),
                position=(x, y),
                train_mode=mode,
                signal_time=signal_time,
                selection_mask_config=config['selection_mask'],
                psf_template=psf_template,
                n_train_splits=config['n_train_splits'],
                base_model_creator=base_model_creator,
                expected_signal=expected_signal,
            )
            residuals[:, x, y] = residuals_for_position
    print()

    # -------------------------------------------------------------------------
    # STEP 2: Compute signal estimate
    # -------------------------------------------------------------------------

    # Compute final signal estimate
    print('Computing signal estimate...', end=' ', flush=True)
    signal_estimate = derotate_combine(
        stack=residuals, parang=parang, mask=~roi_mask
    )
    print('Done!', flush=True)

    # Save the final signal estimate
    print('Saving signal estimate to FITS...', end=' ', flush=True)
    file_path = results_dir / 'signal_estimate.fits'
    save_fits(array=signal_estimate, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
