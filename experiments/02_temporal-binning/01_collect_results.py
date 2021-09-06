"""
Collect results (= SNR, FPF, ...) for PCA or HSR experiments.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import List

import argparse
import os
import time

from astropy.units import Quantity
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import pandas as pd

from hsr4hci.config import load_config
from hsr4hci.data import load_planets, load_psf_template, load_metadata
from hsr4hci.fits import read_fits
from hsr4hci.general import flatten_nested_dict
from hsr4hci.metrics import compute_metrics
from hsr4hci.psf import get_psf_fwhm
from hsr4hci.units import InstrumentUnitsContext


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_pca_results(
    experiment_dir: Path,
    instrument_units_context: InstrumentUnitsContext,
) -> List[dict]:
    """
    Auxiliary function to collect the results for a PCA experiment.
    """

    # Read config file for this experiment and get binning factor
    config = load_config(experiment_dir / 'config.json')
    binning_factor = config['dataset']['binning_factor']

    # Load the FITS file with all signal estimates
    try:
        file_path = experiment_dir / 'results' / 'signal_estimates.fits'
        signal_estimates = read_fits(file_path, return_header=False)
    except FileNotFoundError:
        print('Failed!', flush=True)
        return [{}]

    # Get expected position of the planet (in polar coordinates)
    planet_parameters = load_planets(**config['dataset'])[planet]
    planet_position = (
        Quantity(planet_parameters['separation'], 'arcsec'),
        Quantity(planet_parameters['position_angle'], 'degree'),
    )

    # Store the result for the different number of components
    result = []

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
            results_dict['binning_factor'] = binning_factor
            results_dict['n_components'] = n_components
            result.append(results_dict)

    return result


def get_hsr_results(
    experiment_dir: Path,
    instrument_units_context: InstrumentUnitsContext,
    psf_fwhm: float,
) -> dict:
    """
    Auxiliary function to collect the results for an HSR experiment.
    """

    # Read config file for this experiment and get binning factor
    config = load_config(experiment_dir / 'config.json')
    binning_factor = config['dataset']['binning_factor']

    # Load the FITS file with all signal estimates
    try:
        file_path = experiment_dir / 'results' / 'signal_estimate.fits'
        signal_estimate = read_fits(file_path, return_header=False)
    except FileNotFoundError:
        return {}

    # Get expected position of the planet (in polar coordinates)
    planet_parameters = load_planets(**config['dataset'])[planet]
    planet_position = (
        Quantity(planet_parameters['separation'], 'arcsec'),
        Quantity(planet_parameters['position_angle'], 'degree'),
    )

    # Compute the metrics (SNR, FPF, ...), add binning factor to the
    # flattened result dictionary, and store it
    with instrument_units_context:
        tmp_results_dict, _ = compute_metrics(
            frame=signal_estimate,
            polar_position=planet_position,
            aperture_radius=Quantity(psf_fwhm / 2, 'pixel'),
            planet_mode='FS',
            noise_mode='P',
            search_radius=Quantity(1, 'pixel'),
            exclusion_angle=None,
        )
        results_dict = flatten_nested_dict(tmp_results_dict)
        results_dict = {**{'binning_factor': binning_factor}, **results_dict}

    return results_dict


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCOLLECT RESULTS FOR TEMPORAL BINNING\n', flush=True)

    # -------------------------------------------------------------------------
    # Set up parser and get command line arguments
    # -------------------------------------------------------------------------

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algorithm',
        type=str,
        required=True,
        choices=['pca', 'signal_fitting', 'signal_masking'],
        help='Algorithm: "pca", "signal_fitting" or "signal_masking".',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset, e.g., "beta_pictoris__lp".',
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=8,
        help='Number of parallel jobs for processing the results.',
    )
    parser.add_argument(
        '--planet',
        type=str,
        default='b',
        help='Planet, e.g., "b".',
    )
    args = parser.parse_args()

    # Get arguments
    algorithm = args.algorithm
    dataset = args.dataset
    n_jobs = args.n_jobs
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
    # Loop over experiments and collect the results
    # -------------------------------------------------------------------------

    # Define main directory and collect experiment directories in there
    main_dir = Path(__file__).resolve().parent / dataset / algorithm
    experiment_dirs = [
        Path(_.path) for _ in os.scandir(main_dir) if _.is_dir()
    ]

    # Loop over the experiment directories and collect the SNR, FPF, ...
    print('Collecting results (in parallel):')
    if algorithm == 'pca':
        results = Parallel(n_jobs=n_jobs)(
            delayed(get_pca_results)(_, instrument_units_context)
            for _ in tqdm(experiment_dirs, ncols=80)
        )
        results = sum(results, [])
    elif algorithm in ('signal_fitting', 'signal_masking'):
        results = Parallel(n_jobs=n_jobs)(
            delayed(get_hsr_results)(_, instrument_units_context, psf_fwhm)
            for _ in tqdm(experiment_dirs, ncols=80)
        )
    else:
        raise ValueError(f'Invalid algorithm: {algorithm}')

    # Drop empty results (from experiments that were skipped)
    results = [_ for _ in results if _ != {}]

    # -------------------------------------------------------------------------
    # Convert results to pandas data frame and save to TSV
    # -------------------------------------------------------------------------

    print('\nSaving results to TSV...', end=' ', flush=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(main_dir / f'metrics__{planet}.tsv', sep='\t')
    print('Done', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
