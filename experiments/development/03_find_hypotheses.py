"""
For each spatial position, find the best hypothesis for when the planet
passes through this position.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import os
import time

from astropy.units import Quantity
from tqdm.auto import tqdm

import numpy as np

from hsr4hci.utils.config import load_config
from hsr4hci.utils.data import load_data
from hsr4hci.utils.fits import save_fits
from hsr4hci.utils.hdf import load_dict_from_hdf
from hsr4hci.utils.hypotheses import find_hypothesis
from hsr4hci.utils.masking import get_roi_mask, get_positions_from_mask
from hsr4hci.utils.psf import crop_psf_template
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
    hypotheses_dir = results_dir / 'hypotheses'
    hypotheses_dir.mkdir(exist_ok=True)

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    config = load_config(experiment_dir / 'config.json')
    print('Done!', flush=True)

    # Define shortcuts for entries in the experiment config
    max_signal_length = config['signal_masking']['max_signal_length']
    metric_function = config['signal_masking']['metric_function']
    n_signal_times = config['signal_masking']['n_signal_times']

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

    # Define various useful shortcuts
    frame_size = stack.shape[1:]
    pixscale = float(metadata['PIXSCALE'])
    lambda_over_d = float(metadata['LAMBDA_OVER_D'])

    # -------------------------------------------------------------------------
    # Activate the unit conversions for this instrument
    # -------------------------------------------------------------------------

    set_units_for_instrument(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
        verbose=False,
    )

    # -------------------------------------------------------------------------
    # Crop the PSF template to a region with radius 1 lambda / D
    # -------------------------------------------------------------------------

    print('Cropping PSF template...', end=' ', flush=True)
    psf_cropped = crop_psf_template(
        psf_template=psf_template, psf_radius=Quantity(1.0, 'lambda_over_d')
    )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Set up the mask for the region of interest (ROI)
    # -------------------------------------------------------------------------

    # Define a mask for the ROI
    roi_mask = get_roi_mask(
        mask_size=frame_size,
        inner_radius=Quantity(*config['roi_mask']['inner_radius']),
        outer_radius=Quantity(*config['roi_mask']['outer_radius']),
    )

    # -------------------------------------------------------------------------
    # Find best hypothesis for every spatial position and save results to FITS
    # -------------------------------------------------------------------------

    # Initialize hypotheses array
    hypotheses = np.full(frame_size, np.nan)

    # Loop over all spatial positions and find the respective hypothesis
    print('\nFinding best hypothesis for each spatial pixel:', flush=True)
    for position in tqdm(get_positions_from_mask(roi_mask), ncols=80):

        hypotheses[position[0], position[1]] = find_hypothesis(
            results=results,
            position=position,
            parang=parang,
            n_signal_times=n_signal_times,
            frame_size=frame_size,
            psf_cropped=psf_cropped,
            max_signal_length=max_signal_length,
            metric_function=metric_function,
        )

    # Save hypotheses as a FITS file
    print('\nSaving hypotheses to FITS...', end=' ', flush=True)
    file_path = hypotheses_dir / f'hypotheses__{metric_function}.fits'
    save_fits(array=hypotheses, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
