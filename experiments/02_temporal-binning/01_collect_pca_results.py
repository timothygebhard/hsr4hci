"""
Get results (= SNR) for PCA experiments. This includes computing the
SNR for different numbers of components.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import time

from astropy.units import Quantity
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from hsr4hci.config import load_config, get_hsr4hci_dir
from hsr4hci.coordinates import polar2cartesian
from hsr4hci.fits import read_fits
from hsr4hci.data import load_planets, load_psf_template, load_metadata
from hsr4hci.evaluation import compute_optimized_snr
from hsr4hci.psf import get_psf_fwhm
from hsr4hci.units import set_units_for_instrument


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nGET PCA RESULTS\n', flush=True)

    # -------------------------------------------------------------------------
    # Set up parser and get command line arguments
    # -------------------------------------------------------------------------

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['beta_pictoris__lp', 'r_cra__lp'],
        required=True,
    )
    parser.add_argument(
        '--planet',
        type=str,
        default='b',
    )
    args = parser.parse_args()

    # Get arguments
    dataset = args.dataset
    planet = args.planet

    # -------------------------------------------------------------------------
    # Load metadata and PSF template
    # -------------------------------------------------------------------------

    # Load the metadata and set up the unit conversions
    print('Loading metadata and setting up units...', end=' ', flush=True)
    metadata = load_metadata(name=dataset)
    set_units_for_instrument(
        pixscale=Quantity(metadata['PIXSCALE'], 'arcsec / pix'),
        lambda_over_d=Quantity(metadata['LAMBDA_OVER_D'], 'arcsec'),
        verbose=False,
    )
    print('Done!', flush=True)

    # Load the PSF template and estimate its FWHM
    print('Loading PSF template...', end=' ', flush=True)
    psf_template = load_psf_template(name=dataset).squeeze()
    psf_fwhm = round(get_psf_fwhm(psf_template), 2)
    print(f'Done! (psf_radius = {psf_fwhm})', flush=True)

    # -------------------------------------------------------------------------
    # Compute SNRs for each binning factor
    # -------------------------------------------------------------------------

    # Define main directory (that holds experiment directories)
    main_dir = (
        get_hsr4hci_dir()
        / 'experiments'
        / '02_temporal-binning'
        / dataset
        / 'pca'
    )

    # Define binning factors
    factors = (
        1, 2, 3, 4, 5, 6, 8, 10, 16, 25, 32, 64, 128, 150, 200, 300, 400, 500
    )

    # Keep track of the results
    results = []

    # Loop over the binning factors to compute the results
    for factor in factors:

        print(f'Computing SNRs for factor = {factor}:', flush=True)

        # Define path to experiment directory
        experiment_dir = main_dir / f'factor_{factor}'

        # Load experiment config
        config = load_config(experiment_dir / 'config.json')

        # Load information about the planets in the dataset
        planets = load_planets(**config['dataset'])

        # Load the FITS file with all signal estimates
        file_path = experiment_dir / 'results' / 'signal_estimates.fits'
        signal_estimates = np.asarray(read_fits(file_path))
        frame_size = signal_estimates.shape[1:]

        # Get planet parameters and position
        parameters = planets[planet]
        planet_position = polar2cartesian(
            separation=Quantity(parameters['separation'], 'arcsec'),
            angle=Quantity(parameters['position_angle'], 'degree'),
            frame_size=frame_size,
        )

        # Loop over a "reasonable" range of principal components
        for n_pc in tqdm(range(1, 51), ncols=80):

            idx = n_pc - int(config['pca']['min_n'])

            # Compute the figures of merit
            try:
                ignore_neighbors = 1
                results_dict = compute_optimized_snr(
                    frame=signal_estimates[idx],
                    position=planet_position,
                    aperture_radius=Quantity(psf_fwhm / 2, 'pixel'),
                    ignore_neighbors=ignore_neighbors,
                )
            except ValueError:
                ignore_neighbors = 0
                results_dict = compute_optimized_snr(
                    frame=signal_estimates[idx],
                    position=planet_position,
                    aperture_radius=Quantity(psf_fwhm / 2, 'pixel'),
                    ignore_neighbors=ignore_neighbors,
                )

            # Store the results for the current combination of binning factor
            # and number of principal components
            results.append(
                dict(
                    factor=factor,
                    n_pc=n_pc,
                    snr=results_dict['snr'],
                )
            )

    # Convert the results to a pandas data frame and save as a TSV file
    print('Saving results to TSV...', end=' ', flush=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(main_dir / f'snr__{planet}.tsv', sep='\t')
    print('\nDone', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
