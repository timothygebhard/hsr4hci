"""
Merge partial FITS files for the hypotheses and the match fractions
into single FITS files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import os
import time

from astropy.units import Quantity

import h5py
import numpy as np

from hsr4hci.config import load_config
from hsr4hci.data import load_metadata, load_parang, load_psf_template
from hsr4hci.derotating import derotate_combine
from hsr4hci.fits import read_fits, save_fits
from hsr4hci.masking import get_roi_mask
from hsr4hci.residuals import (
    assemble_residual_stack_from_hypotheses,
    get_residual_selection_mask,
)
from hsr4hci.units import InstrumentUnitsContext


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nGET SIGNAL ESTIMATE\n', flush=True)

    # -------------------------------------------------------------------------
    # Set up parser to get command line arguments; get experiment directory
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

    # Get experiment directory
    experiment_dir = Path(os.path.expanduser(args.experiment_dir))
    if not experiment_dir.exists():
        raise NotADirectoryError(f'{experiment_dir} does not exist!')

    # -------------------------------------------------------------------------
    # Load experiment configuration; define unit conversion context
    # -------------------------------------------------------------------------

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    config = load_config(experiment_dir / 'config.json')
    print('Done!', flush=True)

    # Load frames, parallactic angles, etc. from HDF file
    print('Loading data set...', end=' ', flush=True)
    parang = load_parang(**config['dataset'])
    psf_template = load_psf_template(**config['dataset'])
    metadata = load_metadata(**config['dataset'])
    print('Done!', flush=True)

    # Normalize the PSF template
    psf_template /= np.max(psf_template)

    # Define the unit conversion context for this data set
    instrument_units_context = InstrumentUnitsContext(
        pixscale=Quantity(metadata['PIXSCALE'], 'arcsec / pixel'),
        lambda_over_d=Quantity(metadata['LAMBDA_OVER_D'], 'arcsec'),
    )

    # -------------------------------------------------------------------------
    # Load match fraction and construct selection mask
    # -------------------------------------------------------------------------

    # Load the match fraction from FITS
    print('Loading match fraction from FITS...', end=' ', flush=True)
    file_path = experiment_dir / 'match_fractions' / 'median_mf.fits'
    match_fraction = read_fits(file_path, return_header=False)
    frame_size = (match_fraction.shape[0], match_fraction.shape[1])
    print('Done!', flush=True)

    # Construct the mask for the region of interest (ROI)
    with instrument_units_context:
        roi_mask = get_roi_mask(
            mask_size=frame_size,
            inner_radius=Quantity(*config['roi_mask']['inner_radius']),
            outer_radius=Quantity(*config['roi_mask']['outer_radius']),
        )

    # Construct selection mask
    print('\nComputing selection mask...', end=' ', flush=True)
    selection_mask, _, _, _, _ = get_residual_selection_mask(
        match_fraction=match_fraction,
        parang=parang,
        psf_template=psf_template,
    )
    print('Done!', flush=True)

    # Make sure the results directory exists
    results_dir = experiment_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    # Store selection mask as FITS (add threshold to header). Note that FITS
    # does not support boolean masks, so we need to convert to integer first.
    print('Saving selection mask to FITS...', end=' ', flush=True)
    file_path = results_dir / 'selection_mask.fits'
    save_fits(array=selection_mask.astype(int), file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Construct residuals based on selection mask and hypotheses
    # -------------------------------------------------------------------------

    # Load hypotheses from FITS
    print('\nLoading hypotheses from FITS...', end=' ', flush=True)
    file_path = experiment_dir / 'hypotheses' / 'hypotheses.fits'
    hypotheses = read_fits(file_path, return_header=False)
    print('Done!', flush=True)

    # Load residuals from HDF (based on hypotheses and selection mask)
    print('Constructing residuals...', end=' ', flush=True)
    file_path = experiment_dir / 'hdf' / 'residuals.hdf'
    with h5py.File(file_path, 'r') as residuals:
        residual_stack = assemble_residual_stack_from_hypotheses(
            residuals=residuals,
            hypotheses=hypotheses,
            selection_mask=selection_mask,
        )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Compute signal estimate by derotating and averaging the residuals
    # -------------------------------------------------------------------------

    # Compute signal estimate
    print('Computing signal estimate...', end=' ', flush=True)
    signal_estimate = derotate_combine(
        stack=residual_stack, parang=parang, mask=~roi_mask
    )
    print('Done!', flush=True)

    print('Saving signal estimate to FITS...', end=' ', flush=True)
    file_path = results_dir / 'signal_estimate.fits'
    save_fits(array=signal_estimate, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
