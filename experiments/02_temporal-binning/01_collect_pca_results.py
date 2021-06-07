"""
Collect results (= SNR, FPF, ...) for PCA experiments.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import os
import time

from astropy.units import Quantity

import pandas as pd

from hsr4hci.config import load_config, get_hsr4hci_dir
from hsr4hci.data import load_planets, load_psf_template, load_metadata
from hsr4hci.fits import read_fits
from hsr4hci.general import flatten_nested_dict
from hsr4hci.metrics import compute_metrics
from hsr4hci.psf import get_psf_fwhm
from hsr4hci.units import InstrumentUnitsContext


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCOLLECT PCA RESULTS\n', flush=True)

    # -------------------------------------------------------------------------
    # Set up parser and get command line arguments
    # -------------------------------------------------------------------------

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
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
    metadata = load_metadata(name_or_path=dataset)
    instrument_units_context = InstrumentUnitsContext(
        pixscale=Quantity(metadata['PIXSCALE'], 'arcsec / pix'),
        lambda_over_d=Quantity(metadata['LAMBDA_OVER_D'], 'arcsec'),
    )
    print('Done!', flush=True)

    # Load the PSF template and estimate its FWHM
    print('Loading PSF template...', end=' ', flush=True)
    psf_template = load_psf_template(name_or_path=dataset)
    psf_fwhm = get_psf_fwhm(psf_template)
    print(f'Done! (psf_radius = {psf_fwhm:.2f})\n', flush=True)

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

    # Find factors by filtering folders in the main directory
    factors = sorted(
        [
            int(name.split('_')[1])
            for name in filter(
                lambda _: os.path.isdir(main_dir / _), os.listdir(main_dir)
            )
        ]
    )

    # Keep track of the results
    results = []

    # Loop over the binning factors to compute the results
    for factor in factors:

        print(
            f'Computing metrics for factor = {factor}...', end=' ', flush=True
        )

        # Define path to experiment directory; load experiment config
        experiment_dir = main_dir / f'factor_{factor}'
        config = load_config(experiment_dir / 'config.json')

        # Load the FITS file with all signal estimates
        try:
            file_path = experiment_dir / 'results' / 'signal_estimates.fits'
            signal_estimates = read_fits(file_path, return_header=False)
        except FileNotFoundError:
            print('Failed!', flush=True)
            continue

        # Get expected position of the planet (in polar coordinates)
        planet_parameters = load_planets(**config['dataset'])[planet]
        planet_position = (
            Quantity(planet_parameters['separation'], 'arcsec'),
            Quantity(planet_parameters['position_angle'], 'degree'),
        )

        # Loop over different numbers of principal components
        for n_components in (1, 5, 10, 20, 50, 100):

            # Compute the metrics (SNR, FPF, ...), add binning factor and the
            # number of PCs to the flattened result dictionary, and store it
            with instrument_units_context:
                tmp_results_dict, _ = compute_metrics(
                    frame=signal_estimates[n_components - 1],
                    polar_position=planet_position,
                    aperture_radius=Quantity(psf_fwhm / 2, 'pixel'),
                    planet_mode='FS',
                    noise_mode='P',
                    search_radius=Quantity(1, 'pixel'),
                    exclusion_angle=None,
                )
                results_dict = flatten_nested_dict(tmp_results_dict)
                results_dict['factor'] = factor
                results_dict['n_components'] = n_components
                results.append(results_dict)

        print('Done!', flush=True)

    # Convert the results to a pandas data frame and save as a TSV file
    print('\nSaving results to TSV...', end=' ', flush=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(main_dir / f'metrics__{planet}.tsv', sep='\t')
    print('Done', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
