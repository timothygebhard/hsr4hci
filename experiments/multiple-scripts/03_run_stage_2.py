"""
Run stage 2 of the pipeline (find hypotheses, compute MF, ...)
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import os
import time

from astropy.units import Quantity

import numpy as np

from hsr4hci.utils.config import load_config
from hsr4hci.utils.consistency_checks import get_match_fraction
from hsr4hci.utils.data import load_data
from hsr4hci.utils.fits import save_fits
from hsr4hci.utils.hdf import load_dict_from_hdf
from hsr4hci.utils.hypotheses import get_all_hypotheses
from hsr4hci.utils.masking import get_roi_mask
from hsr4hci.utils.signal_estimates import get_signal_estimate
from hsr4hci.utils.signal_masking import assemble_residuals_from_hypotheses
from hsr4hci.utils.units import set_units_for_instrument


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nRUN STAGE 2: FIND HYPOTHESES, COMPUTE MF, ...\n', flush=True)

    # -------------------------------------------------------------------------
    # Load experiment configuration and data; parse command line arguments
    # -------------------------------------------------------------------------

    # Define paths for experiment folder and results folder
    experiment_dir = Path(os.path.realpath(__file__)).parent
    results_dir = experiment_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    config = load_config(experiment_dir / 'config.json')
    print('Done!', flush=True)

    # Load frames, parallactic angles, etc. from HDF file
    print('Loading data set...', end=' ', flush=True)
    stack, parang, psf_template, observing_conditions, metadata = load_data(
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
    n_signal_times = config['n_signal_times']

    # Activate the unit conversions for this instrument
    set_units_for_instrument(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
        verbose=False,
    )

    # Construct the mask for the region of interest (ROI)
    roi_mask = get_roi_mask(
        mask_size=frame_size,
        inner_radius=Quantity(*config['roi_mask']['inner_radius']),
        outer_radius=Quantity(*config['roi_mask']['outer_radius']),
    )

    # -------------------------------------------------------------------------
    # STEP 1: Load results after (parallel) training
    # -------------------------------------------------------------------------

    file_path = results_dir / 'results.hdf'
    results = load_dict_from_hdf(file_path=file_path)

    # -------------------------------------------------------------------------
    # STEP 2: Find hypotheses
    # -------------------------------------------------------------------------

    # Find best hypothesis for every pixel
    print('\nFinding best hypothesis for each spatial pixel:', flush=True)
    hypotheses, similarities = get_all_hypotheses(
        roi_mask=roi_mask,
        results=results,
        parang=parang,
        n_signal_times=n_signal_times,
        frame_size=frame_size,
        psf_template=psf_template,
    )

    # Create directory for hypothesis
    hypotheses_dir = results_dir / 'hypotheses'
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

    # Compute matches for every pixel
    print('\nComputing matches:', flush=True)
    match_fraction__mean, match_fraction__median, _ = get_match_fraction(
        results=results,
        hypotheses=hypotheses,
        parang=parang,
        psf_template=psf_template,
    )

    # Create matches directory
    matches_dir = results_dir / 'matches'
    matches_dir.mkdir(exist_ok=True)

    # Save match fraction(s) as FITS file
    print('Saving match fraction to FITS...', end=' ', flush=True)
    file_path = matches_dir / 'match_fraction__mean.fits'
    save_fits(array=match_fraction__mean, file_path=file_path)
    file_path = matches_dir / 'match_fraction__median.fits'
    save_fits(array=match_fraction__median, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # STEP 4: Threshold match fraction and construct signal estimate
    # -------------------------------------------------------------------------

    # Select residuals for default case and for signal masking / fitting
    print('\nAssembling residuals...', end=' ', flush=True)
    default_residuals = results['default']['residuals']
    non_default_residuals = assemble_residuals_from_hypotheses(
        hypotheses=hypotheses, results=results
    )
    print('Done!', flush=True)

    # Threshold the match fraction, apply morphological filter, and use the
    # resulting mask to construct the final signal estimate
    print('\nComputing signal estimates...', end=' ', flush=True)
    signal_estimate, selection_mask, threshold = get_signal_estimate(
        parang=parang,
        match_fraction=match_fraction__median,
        default_residuals=default_residuals,
        non_default_residuals=non_default_residuals,
        filter_size=int(config['consistency_checks']['filter_size']),
        roi_mask=roi_mask,
    )
    print(f'Done! (threshold = {threshold:.3f})', flush=True)

    print('Saving selection_mask mask to FITS...', end=' ', flush=True)
    array = np.array(selection_mask).astype(int)
    file_path = results_dir / 'selection_mask.fits'
    save_fits(array=array, file_path=file_path)
    print('Done!', flush=True)

    print('Saving signal estimate to FITS...', end=' ', flush=True)
    array = np.array(signal_estimate)
    file_path = results_dir / 'signal_estimate.fits'
    save_fits(array=array, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
