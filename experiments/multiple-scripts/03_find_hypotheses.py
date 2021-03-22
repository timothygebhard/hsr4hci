"""
For each spatial position, find the best hypothesis for the time when
a planet passes through this position.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import os
import time

from astropy.units import Quantity

from hsr4hci.utils.config import load_config
from hsr4hci.utils.data import load_data
from hsr4hci.utils.fits import save_fits
from hsr4hci.utils.hdf import load_dict_from_hdf
from hsr4hci.utils.hypotheses import get_all_hypotheses
from hsr4hci.utils.masking import get_roi_mask
from hsr4hci.utils.units import set_units_for_instrument


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nFIND HYPOTHESES\n', flush=True)

    # -------------------------------------------------------------------------
    # Load experiment configuration
    # -------------------------------------------------------------------------

    # Define paths for experiment folder and results folder
    experiment_dir = Path(os.path.realpath(__file__)).parent
    results_dir = experiment_dir / 'results'

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    config = load_config(experiment_dir / 'config.json')
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Load data set and results, define shortcuts
    # -------------------------------------------------------------------------

    # Load frames, parallactic angles, etc. from HDF file
    print('Loading data set...', end=' ', flush=True)
    stack, parang, psf_template, observing_conditions, metadata = load_data(
        **config['dataset']
    )
    print('Done!', flush=True)

    # Load results from HDF file
    print('Loading main results file...', end=' ', flush=True)
    file_path = results_dir / 'results.hdf'
    results = load_dict_from_hdf(file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Define shortcuts; activate the unit conversions; set up ROI mask
    # -------------------------------------------------------------------------

    # Define various useful shortcuts
    frame_size = stack.shape[1:]
    pixscale = float(metadata['PIXSCALE'])
    lambda_over_d = float(metadata['LAMBDA_OVER_D'])

    # Activate the unit conversions for this instrument
    set_units_for_instrument(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
        verbose=False,
    )

    # Set up a mask for the ROI
    roi_mask = get_roi_mask(
        mask_size=frame_size,
        inner_radius=Quantity(*config['roi_mask']['inner_radius']),
        outer_radius=Quantity(*config['roi_mask']['outer_radius']),
    )

    # -------------------------------------------------------------------------
    # Find best hypothesis for every spatial position and save results to FITS
    # -------------------------------------------------------------------------

    # Loop over all spatial positions and find the respective hypothesis
    print('\nFinding best hypothesis for each spatial pixel:', flush=True)
    hypotheses = get_all_hypotheses(
        roi_mask=roi_mask,
        results=results,
        parang=parang,
        n_signal_times=config['signal_masking']['n_signal_times'],
        frame_size=frame_size,
        psf_template=psf_template,
        max_signal_length=config['signal_masking']['max_signal_length'],
        metric_function=config['signal_masking']['metric_function'],
    )

    # Set up hypotheses directory
    hypotheses_dir = results_dir / 'hypotheses'
    hypotheses_dir.mkdir(exist_ok=True)

    # Save hypotheses as a FITS file
    print('\nSaving hypotheses to FITS...', end=' ', flush=True)
    file_path = hypotheses_dir / 'hypotheses.fits'
    save_fits(array=hypotheses, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
