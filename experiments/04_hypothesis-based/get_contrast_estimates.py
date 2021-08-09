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

import numpy as np

import pandas as pd

from hsr4hci.config import load_config
from hsr4hci.coordinates import polar2cartesian, cartesian2polar
from hsr4hci.data import load_metadata, load_psf_template
from hsr4hci.fits import read_fits
from hsr4hci.photometry import (
    get_stellar_flux,
    get_fluxes_for_polar_positions,
    get_flux,
)
from hsr4hci.psf import get_psf_fwhm
from hsr4hci.positions import get_reference_positions
from hsr4hci.units import (
    flux_ratio_to_magnitudes,
    magnitude_to_flux_ratio,
    InstrumentUnitsContext,
)


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
    # Loop over different experiments to collect contrast values
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
        # STEP 1: Preliminaries
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
        # STEP 2: Fit the PSF size and measure the stellar flux
        # ---------------------------------------------------------------------

        # Fit the FWHM of the PSF
        psf_fwhm = get_psf_fwhm(psf_template)

        # Measure the stellar flux
        stellar_flux = get_stellar_flux(
            psf_template=psf_template,
            dit_stack=float(metadata['DIT_STACK']),
            dit_psf_template=float(metadata['DIT_PSF_TEMPLATE']),
            scaling_factor=float(metadata['ND_FILTER']),
            mode='FS',
        )

        # ---------------------------------------------------------------------
        # STEP 3: Run photometry for planet (only use planet "b" for now)
        # ---------------------------------------------------------------------

        # Get the assumed position of the planet
        initial_position_polar = (
            Quantity(*config['hypothesis']['b']['separation']),
            Quantity(*config['hypothesis']['b']['position_angle']),
        )

        # Compute the raw planet flux and optimize the position
        with instrument_unit_context:
            final_position, raw_planet_flux = get_flux(
                frame=np.nan_to_num(signal_estimate),
                position=polar2cartesian(
                    *initial_position_polar, frame_size=frame_size
                ),
                mode='FS',
                aperture_radius=Quantity(psf_fwhm / 2, 'pixel'),
                search_radius=Quantity(1, 'pixel'),
            )

        # Compute reference positions based on final position
        with instrument_unit_context:
            reference_positions = get_reference_positions(
                polar_position=cartesian2polar(final_position, frame_size),
                aperture_radius=Quantity(psf_fwhm / 2, 'pixel'),
                exclusion_angle=None,
            )

        # Measure the flux at the reference positions
        reference_fluxes = get_fluxes_for_polar_positions(
            polar_positions=reference_positions,
            frame=signal_estimate,
            mode='P',
        )

        # Compute the "background-corrected" flux estimate
        planet_flux = raw_planet_flux - float(np.mean(reference_fluxes))

        # Compute brightness ratio of the star and the planet in magnitudes;
        # we call this the observed contrast
        flux_ratio_observed = planet_flux / stellar_flux
        contrast_observed = flux_ratio_to_magnitudes(flux_ratio_observed)

        # Convert the expected contrast (from the literature) to a flux ratio
        contrast_expected = config['hypothesis']['b']['contrast']
        flux_ratio_expected = magnitude_to_flux_ratio(contrast_expected)

        # Compute "throughput" (ratio of our magnitude vs. true magnitude)
        throughput = flux_ratio_observed / flux_ratio_expected

        # Store all results
        results.append(
            {
                'train_mode': train_mode,
                'binning_factor': binning_factor,
                'dataset': dataset,
                'stellar_flux': stellar_flux,
                'raw_planet_flux': raw_planet_flux,
                'planet_flux': planet_flux,
                'mean_reference_flux': float(np.mean(reference_fluxes)),
                'flux_ratio_observed': flux_ratio_observed,
                'flux_ratio_expected': flux_ratio_expected,
                'contrast_observed': contrast_observed,
                'contrast_expected': contrast_expected,
                'throughput': throughput,
                'label': f'{contrast_observed:.2f} ({throughput:.2f})',
            }
        )

        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # STEP 5: Convert results to data frame and save them as a TSV
    # -------------------------------------------------------------------------

    # Convert to data frame
    results_df = pd.DataFrame(results)

    # Print results
    print('\n')
    print(results_df)

    # Store results as TSV file
    print('\nSaving results to TSV...', end=' ', flush=True)
    file_path = Path('.') / 'results.tsv'
    results_df.to_csv(file_path, sep='\t')
    print('Done!', flush=True)

    # Create pivot table and save LaTeX code
    pivot_table = pd.pivot_table(
        results_df,
        values="label",
        index=["train_mode"],
        columns=["dataset", "binning_factor"],
        aggfunc=lambda x: x,
    )
    print()
    print(pivot_table)
    print()
    latex_code = pivot_table.to_latex()
    with open('results-table.tex', 'w') as tex_file:
        tex_file.write(latex_code)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
