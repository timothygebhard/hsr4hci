"""
Run consistency checks on hypotheses and compute match fraction.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import os
import time

from astropy.units import Quantity

from hsr4hci.utils.config import load_config
from hsr4hci.utils.consistency_checks import get_match_fraction
from hsr4hci.utils.data import load_data
from hsr4hci.utils.fits import read_fits, save_fits
from hsr4hci.utils.hdf import load_dict_from_hdf
from hsr4hci.utils.units import set_units_for_instrument


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCOMPUTE MATCHES AND MATCH FRACTION\n', flush=True)

    # -------------------------------------------------------------------------
    # Load experiment configuration and define shortcuts
    # -------------------------------------------------------------------------

    # Define paths for experiment folder and result folders
    experiment_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    results_dir = experiment_dir / 'results'
    hypotheses_dir = results_dir / 'hypotheses'

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    file_path = experiment_dir / 'config.json'
    config = load_config(file_path=file_path)
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

    # Load hypotheses from FITS file
    print('Loading hypotheses file...', end=' ', flush=True)
    file_path = hypotheses_dir / 'hypotheses.fits'
    hypotheses = read_fits(file_path=file_path)
    print('Done!', flush=True)

    # Define various useful shortcuts
    n_frames, x_size, y_size = stack.shape
    frame_size = (x_size, y_size)
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
    # Compute the consistency checks and get the matches for each pixel
    # -------------------------------------------------------------------------

    # Create matches directory
    matches_dir = results_dir / 'matches'
    matches_dir.mkdir(exist_ok=True)

    print('Computing match fraction:', flush=True)
    (
        match_fraction__mean,
        match_fraction__median,
        affected_pixels,
    ) = get_match_fraction(
        results=results,
        hypotheses=hypotheses,
        parang=parang,
        psf_template=psf_template,
        metric_function=config['signal_masking']['metric_function'],
        verbose=True,
    )

    # -------------------------------------------------------------------------
    # Compute the match fraction and save it as a FITS file
    # -------------------------------------------------------------------------

    print('\nSaving match fraction to FITS...', end=' ', flush=True)

    # Save mean match fraction
    file_path = matches_dir / 'match_fraction__mean.fits'
    save_fits(array=match_fraction__mean, file_path=file_path)

    # Save median match fraction
    file_path = matches_dir / 'match_fraction__median.fits'
    save_fits(array=match_fraction__median, file_path=file_path)

    # Save affected pixels
    file_path = matches_dir / 'affected_pixels.fits'
    save_fits(array=affected_pixels, file_path=file_path)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
