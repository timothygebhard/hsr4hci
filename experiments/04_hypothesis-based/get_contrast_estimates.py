"""
Compute estimated contrast from signal estimate.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from itertools import product
from pathlib import Path

import time

from astropy.units import Quantity

import pandas as pd

from hsr4hci.config import load_config
from hsr4hci.contrast import get_contrast
from hsr4hci.data import load_metadata, load_psf_template
from hsr4hci.fits import read_fits
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
    print('\nCOMPUTE ESTIMATED CONTRAST FROM SIGNAL ESTIMATE\n', flush=True)

    # -------------------------------------------------------------------------
    # Loop over different experiments to collect contrast / throughput values
    # -------------------------------------------------------------------------

    # Initialize list for results
    results = []

    # Loop over experiments
    for train_mode, dataset, binning_factor in product(
        ('signal_fitting', 'signal_masking'),
        ('beta_pictoris__lp', 'beta_pictoris__mp', 'r_cra__lp'),
        (1, 10, 100),
    ):

        # ---------------------------------------------------------------------
        # Step 1: Preliminaries
        # ---------------------------------------------------------------------

        # Define experiment directory
        experiment_dir = (
            Path('.')
            / train_mode
            / dataset
            / f'binning-factor_{binning_factor}'
        )

        print(f'Processing {experiment_dir}...', end=' ', flush=True)

        # Load experiment config from JSON
        config = load_config(experiment_dir / 'config.json')

        # Load the PSF template and metadata of the dataset
        psf_template = load_psf_template(**config['dataset'])
        metadata = load_metadata(**config['dataset'])

        # Fit the FWHM of the PSF
        psf_fwhm = get_psf_fwhm(psf_template)

        # Load the signal estimate
        file_path = experiment_dir / 'results' / 'signal_estimate.fits'
        try:
            signal_estimate = read_fits(file_path, return_header=False)
        except FileNotFoundError:
            print('Failed!', flush=True)
            continue
        frame_size = (signal_estimate.shape[0], signal_estimate.shape[1])

        # Define the unit conversion context for this data set
        instrument_unit_context = InstrumentUnitsContext(
            pixscale=Quantity(float(metadata['PIXSCALE']), 'arcsec / pixel'),
            lambda_over_d=Quantity(float(metadata['LAMBDA_OVER_D']), 'arcsec'),
        )

        # ---------------------------------------------------------------------
        # Step 2: Compute the contrast and throughput
        # ---------------------------------------------------------------------

        with instrument_unit_context:

            # Get the assumed position of the planet
            initial_position_polar = (
                Quantity(*config['hypothesis']['b']['separation']),
                Quantity(*config['hypothesis']['b']['position_angle']),
            )

            # Get the expected contrast
            expected_contrast = config['hypothesis']['b']['contrast']['mean']

            # Get the observed contrast
            result = get_contrast(
                signal_estimate=signal_estimate,
                polar_position=initial_position_polar,
                psf_template=psf_template,
                metadata=metadata,
                no_fake_planets=None,
                expected_contrast=expected_contrast,
            )

            # Store all relevant results
            results.append(
                {
                    'train_mode': train_mode,
                    'binning_factor': binning_factor,
                    'dataset': dataset,
                    'observed_flux_ratio': result['observed_flux_ratio'],
                    'observed_contrast': result['observed_contrast'],
                    'expected_flux_ratio': result['expected_flux_ratio'],
                    'expected_contrast': result['expected_contrast'],
                    'throughput': result['throughput'],
                    'label': (
                        f'{result["observed_contrast"]:.2f} '
                        f'({result["throughput"]:.2f})'
                    ),
                }
            )

        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Convert results to data frame and save them as a TSV
    # -------------------------------------------------------------------------

    # Convert to data frame and print it
    results_df = pd.DataFrame(results)
    print('\n', results_df)

    # Store results as TSV file
    print('\nSaving results to TSV...', end=' ', flush=True)
    file_path = Path('.') / 'results.tsv'
    results_df.to_csv(file_path, sep='\t')
    print('Done!', flush=True)

    # Create and print pivot table
    pivot_table = pd.pivot_table(
        results_df,
        values="label",
        index=["train_mode"],
        columns=["dataset", "binning_factor"],
        aggfunc=lambda x: x,
    )
    print('\n\n', pivot_table, '\n')

    # Save pivot table as LaTeX code
    latex_code = pivot_table.to_latex()
    with open('results-table.tex', 'w') as tex_file:
        tex_file.write(latex_code)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
