"""
Collect results for the sub-experiments of an algorithm, that is,
compute the observed contrast, achieved throughput, SNR / FPF, etc.
for every combination of separation, azimuthal position and contrast.
The results are stored in a single TSV file.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Optional

import argparse
import os
import time

from astropy.units import Quantity
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

from hsr4hci.config import load_config
from hsr4hci.contrast import get_contrast
from hsr4hci.data import (
    load_metadata,
    load_psf_template,
)
from hsr4hci.fits import read_fits
from hsr4hci.general import flatten_nested_dict
from hsr4hci.metrics import compute_metrics
from hsr4hci.positions import get_injection_position
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
    print('\nCOLLECT RESULTS\n')

    # -------------------------------------------------------------------------
    # Parse command line arguments
    # -------------------------------------------------------------------------

    # Set up a parser and parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        required=True,
        help='Main directory of the experiment set.',
    )
    parser.add_argument(
        '--mode',
        choices=['classic', 'alternative'],
        default='classic',
        help=(
            'How to compute the throughput: using the "classic" approach'
            'where the no_fake_planets residual is subtracted, or the '
            'alternative approach that estimate the background from other '
            'positions at the same separation.'
        ),
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=32,
        help='Number of parallel jobs for processing the results.',
    )
    args = parser.parse_args()

    # Define shortcuts
    main_dir = Path(args.directory).resolve()
    mode = args.mode
    n_jobs = args.n_jobs

    # Make sure the main directory (where the base_config.json resides) exists
    if not main_dir.exists():
        raise RuntimeError(f'{main_dir} does not exist!')

    # -------------------------------------------------------------------------
    # Load the (base) config, the PSF template and the metadata
    # -------------------------------------------------------------------------

    # Load base configuration
    base_config = load_config(main_dir / 'base_config.json')

    # Load PSF template and metadata
    psf_template = load_psf_template(**base_config['dataset'])
    metadata = load_metadata(**base_config['dataset'])

    # Define shortcuts to metadata of the data set
    pixscale = float(metadata['PIXSCALE'])
    lambda_over_d = float(metadata['LAMBDA_OVER_D'])

    # Define the unit conversion context for this data set
    instrument_units_context = InstrumentUnitsContext(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
    )

    # Get FWHM of the PSF template
    psf_fwhm = get_psf_fwhm(psf_template=psf_template)

    # -------------------------------------------------------------------------
    # Find experiment folders for which we have results
    # -------------------------------------------------------------------------

    # Define directory that holds all the experiment folders
    experiments_dir = main_dir / 'experiments'

    # Get a list of all experiment folders (except no_fake_planets)
    experiment_dirs = list(
        filter(
            lambda _: os.path.isdir(_) and 'no_fake_planets' not in str(_),
            sorted([experiments_dir / _ for _ in os.listdir(experiments_dir)]),
        )
    )

    # -------------------------------------------------------------------------
    # Load the no_fake_planets residual (if applicable)
    # -------------------------------------------------------------------------

    no_fake_planets: Optional[np.ndarray] = None
    if mode == 'classic':
        file_path = (
            experiments_dir
            / 'no_fake_planets'
            / 'results'
            / 'signal_estimate.fits'
        )
        if not file_path.exists():
            raise FileNotFoundError(f'{file_path} does not exist!')
        no_fake_planets = read_fits(file_path, return_header=False)

    # -------------------------------------------------------------------------
    # Loop over experiment folders, get signal estimate and compute throughput
    # -------------------------------------------------------------------------

    def get_result(experiment_dir: Path) -> Optional[dict]:
        """
        Auxiliary function to run the computation of the contrast, the
        throughput and the metrics (SNR / FPF) in parallel using joblib.
        """

        # Load the config to get the injection parameters
        config = load_config(experiment_dir / 'config.json')

        # Load the signal estimate
        file_path = experiment_dir / 'results' / 'signal_estimate.fits'
        try:
            signal_estimate = read_fits(file_path, return_header=False)
        except FileNotFoundError:
            return None

        # Define shortcuts
        separation = config['injection']['separation']
        azimuthal_position = config['injection']['azimuthal_position']
        expected_contrast = config['injection']['contrast']

        # Compute the expected position and contrast (note that we need to
        # convert the separation to units of PSF FWHM first!)
        polar_position = get_injection_position(
            separation=Quantity(separation * psf_fwhm, 'pixel'),
            azimuthal_position=azimuthal_position,
        )

        # Compute the contrast and throughput
        # Drop `expected_contrast` because we add it manually to the result
        # dict (to improve the order of keys for sorting the output TSV file)
        contrast_results = get_contrast(
            signal_estimate=signal_estimate,
            polar_position=polar_position,
            psf_template=psf_template,
            metadata=metadata,
            no_fake_planets=no_fake_planets,
            expected_contrast=expected_contrast,
            planet_mode='FS',
            noise_mode='P',
            exclusion_angle=None,
        )
        contrast_results.pop('expected_contrast', None)

        # Compute metrics (i.e., SNR and FPF); flatten the result dict
        metrics_results, _ = compute_metrics(
            frame=signal_estimate,
            polar_position=polar_position,
            aperture_radius=Quantity(psf_fwhm / 2, 'pixel'),
            planet_mode='FS',
            noise_mode='P',
            exclusion_angle=None,
            n_rotation_steps=10,
        )
        metrics_results = flatten_nested_dict(metrics_results)

        # Combine all results that we want to store
        return {
            'separation': separation,
            'expected_contrast': expected_contrast,
            'azimuthal_position': azimuthal_position,
            **contrast_results,
            **metrics_results,
        }

    # Use joblib to compute the throughputs in parallel; drop all results that
    # were None (e.g., because an experiment did not exist or was skipped)
    print('Computing contrasts and throughputs in parallel:')
    results = Parallel(n_jobs=n_jobs)(
        delayed(get_result)(_) for _ in tqdm(experiment_dirs, ncols=80)
    )
    results = [_ for _ in results if _ is not None]

    # -------------------------------------------------------------------------
    # Store the results to a TSV file
    # -------------------------------------------------------------------------

    print('\nSaving results to TSV file...', end=' ', flush=True)

    # Convert result list to a pandas data frame
    results_df = pd.DataFrame(results).sort_values(
        by=['separation', 'expected_contrast', 'azimuthal_position']
    )

    # Store the results
    file_path = main_dir / f'results__{mode}.tsv'
    results_df.to_csv(file_path, sep='\t')

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
