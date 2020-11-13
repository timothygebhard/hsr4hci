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

import bottleneck as bn

from hsr4hci.utils.config import load_config
from hsr4hci.utils.consistency_checks import get_matches
from hsr4hci.utils.data import load_data
from hsr4hci.utils.fits import read_fits, save_fits
from hsr4hci.utils.hdf import load_dict_from_hdf
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

    # Define shortcuts for entries in the experiment config
    filter_size = config['consistency_checks']['filter_size']
    metric_function = config['signal_masking']['metric_function']
    n_test_positions = config['consistency_checks']['n_test_positions']

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
    file_path = hypotheses_dir / f'hypotheses__{metric_function}.fits'
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
    # Crop the PSF template to a region with radius 1 lambda / D
    # -------------------------------------------------------------------------

    print('Cropping PSF template...', end=' ', flush=True)
    psf_cropped = crop_psf_template(
        psf_template=psf_template, psf_radius=Quantity(1.0, 'lambda_over_d')
    )
    print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Compute the consistency checks and get the matches for each pixel
    # -------------------------------------------------------------------------

    # Create matches directory
    matches_dir = results_dir / 'matches'
    matches_dir.mkdir(exist_ok=True)

    # Compute matches: loop over all spatial pixels, compute the planet path
    # that is implied by the respective entry in the `hypotheses` array, and
    # check for `n_test_positions` along that trajectory if the residuals are
    # consistent with the hypothesis. The resulting `matches` array has shape
    # `(n_test_positions, width, height)`.
    print('Computing matches:', flush=True)
    matches = get_matches(
        results=results,
        hypotheses=hypotheses,
        parang=parang,
        psf_cropped=psf_cropped,
        n_test_positions=n_test_positions,
        metric_function=metric_function,
    )
    print('')

    # Save matches to FITS file
    print('Saving matches to FITS...', end=' ', flush=True)
    file_path = matches_dir / f'matches__{metric_function}.fits'
    save_fits(array=matches, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Compute the match fraction and save it as a FITS file
    # -------------------------------------------------------------------------

    # Compute match fraction by averaging along the test positions axis
    # TODO: Should we use the mean or median here?
    match_fraction = bn.nanmean(matches, axis=0)

    # Save match fraction as FITS file
    print('\nSaving match fraction to FITS...', end=' ', flush=True)
    file_path = matches_dir / f'match_fraction__{metric_function}.fits'
    save_fits(array=match_fraction, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
