"""
Run the HSR pipeline in a single script.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict

import argparse
import os
import time

from astropy.units import Quantity

import numpy as np

from hsr4hci.base_models import BaseModelCreator
from hsr4hci.config import load_config
from hsr4hci.data import load_dataset
from hsr4hci.derotating import derotate_combine
from hsr4hci.fits import save_fits
from hsr4hci.hdf import create_hdf_dir, save_dict_to_hdf
from hsr4hci.hypotheses import get_all_hypotheses
from hsr4hci.masking import get_roi_mask
from hsr4hci.match_fraction import get_all_match_fractions
from hsr4hci.residuals import (
    assemble_residual_stack_from_hypotheses,
    get_residual_selection_mask,
)
from hsr4hci.training import train_all_models
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
    parser.add_argument(
        '--dont-save-intermediate-results',
        action='store_true',
        help='If this flag is set, only the final signal estimate is saved.',
    )
    parser.add_argument(
        '--hdf-location',
        type=str,
        choices=['local', 'work'],
        default='work',
        help='Where to create the HDF directory: locally or on /work.',
    )
    args = parser.parse_args()

    # Define shortcut
    save_intermediate = not args.dont_save_intermediate_results

    # -------------------------------------------------------------------------
    # Load experiment configuration and data set
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
    n_signal_times = config['n_signal_times']

    # Define the unit conversion context for this data set
    instrument_units_context = InstrumentUnitsContext(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
    )

    # Construct the mask for the region of interest (ROI)
    with instrument_units_context:
        roi_mask = get_roi_mask(
            mask_size=frame_size,
            inner_radius=Quantity(*config['roi_mask']['inner_radius']),
            outer_radius=Quantity(*config['roi_mask']['outer_radius']),
        )

    # -------------------------------------------------------------------------
    # STEP 1: Train HSR models
    # -------------------------------------------------------------------------

    # Set up a BaseModelCreator to create instances of our base model
    base_model_creator = BaseModelCreator(**config['base_model'])

    print('\nTraining models:', flush=True)
    with instrument_units_context:
        results = train_all_models(
            roi_mask=roi_mask,
            stack=stack,
            parang=parang,
            psf_template=psf_template,
            obscon_array=observing_conditions.as_array(selected_keys),
            selection_mask_config=config['selection_mask'],
            base_model_creator=base_model_creator,
            n_train_splits=config['n_train_splits'],
            train_mode=config['train_mode'],
            n_signal_times=n_signal_times,
            n_roi_splits=1,
            roi_split=0,
            return_format='full',
        )
    residuals: Dict[str, np.ndarray] = dict(results['residuals'])
    print()

    # Save results to HDF if desired
    if save_intermediate:

        # Create HDF directory (either locally or on /work)
        create_on_work = args.hdf_location == 'work'
        hdf_dir = create_hdf_dir(experiment_dir, create_on_work=create_on_work)

        print('Saving residuals to HDF...', end=' ', flush=True)
        file_path = hdf_dir / 'residuals.hdf'
        save_dict_to_hdf(dictionary=results, file_path=file_path)
        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # STEP 2: Find hypotheses
    # -------------------------------------------------------------------------

    # Find best hypothesis for every pixel
    print('\nFinding best hypothesis for each spatial pixel:', flush=True)
    hypotheses, similarities = get_all_hypotheses(
        roi_mask=roi_mask,
        residuals=residuals,
        parang=parang,
        n_signal_times=n_signal_times,
        frame_size=frame_size,
        psf_template=psf_template,
    )

    # Save hypotheses and similarities, if desired
    if save_intermediate:

        # Create directory for hypothesis
        hypotheses_dir = experiment_dir / 'hypotheses'
        hypotheses_dir.mkdir(exist_ok=True)

        # Save hypotheses as a FITS file
        print('\nSaving hypotheses to FITS...', end=' ', flush=True)
        file_path = hypotheses_dir / 'hypotheses.fits'
        save_fits(array=hypotheses, file_path=file_path)
        print('Done!', flush=True)

        # Save cosine similarities (of hypotheses) as a FITS file
        print('Saving similarities to FITS...', end=' ', flush=True)
        file_path = hypotheses_dir / 'similarities.fits'
        save_fits(array=similarities, file_path=file_path)
        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # STEP 3: Compute match fractions
    # -------------------------------------------------------------------------

    # Compute match fraction for every pixel
    print('\nComputing match fraction:', flush=True)
    mean_mf, median_mf, _ = get_all_match_fractions(
        residuals=residuals,
        hypotheses=hypotheses,
        parang=parang,
        psf_template=psf_template,
        roi_mask=roi_mask,
        frame_size=frame_size,
    )

    # Save match fractions, if desired
    if save_intermediate:

        # Create matches directory
        mf_dir = experiment_dir / 'match_fractions'
        mf_dir.mkdir(exist_ok=True)

        # Save match fraction(s) as FITS file
        print('Saving match fraction to FITS...', end=' ', flush=True)
        file_path = mf_dir / 'mean_mf.fits'
        save_fits(array=mean_mf, file_path=file_path)
        file_path = mf_dir / 'median_mf.fits'
        save_fits(array=median_mf, file_path=file_path)
        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # STEP 4: Find selection mask
    # -------------------------------------------------------------------------

    # Compute the selection mask that determines which residual type (default
    # or based on signal fitting / masking) is used for a pixel
    print('\nComputing selection mask for residuals...', end=' ', flush=True)
    selection_mask, _, _, _, _ = get_residual_selection_mask(
        match_fraction=mean_mf,
        parang=parang,
        psf_template=psf_template,
    )
    print('Done!', flush=True)

    # Create results directory
    results_dir = experiment_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    # (Always) save the selection mask
    print('Saving selection_mask mask to FITS...', end=' ', flush=True)
    array = np.array(selection_mask).astype(int)
    file_path = results_dir / 'selection_mask.fits'
    save_fits(array=array, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # STEP 5: Assemble residual stack and compute signal estimate
    # -------------------------------------------------------------------------

    print('Assembling residual stack...', end=' ', flush=True)
    residual_stack = assemble_residual_stack_from_hypotheses(
        residuals=residuals,
        hypotheses=hypotheses,
        selection_mask=selection_mask,
    )
    print('Done!', flush=True)

    # Compute final signal estimate
    print('Computing signal estimate...', end=' ', flush=True)
    signal_estimate = derotate_combine(
        stack=residual_stack, parang=parang, mask=~roi_mask
    )
    print('Done!', flush=True)

    # (Always) save the final signal estimate
    print('Saving signal estimate to FITS...', end=' ', flush=True)
    file_path = results_dir / 'signal_estimate.fits'
    save_fits(array=signal_estimate, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
