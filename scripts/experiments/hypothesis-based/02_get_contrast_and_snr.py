"""
Get the contrast (and SNR) for a hypothesis-based experiment.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import os
import time

from astropy.units import Quantity

import numpy as np
import pandas as pd

from hsr4hci.apertures import (
    get_aperture_flux,
    get_reference_aperture_positions,
)
from hsr4hci.config import load_config
from hsr4hci.contrast_curves import (
    flux_ratio_to_magnitudes,
    magnitude_to_flux_ratio,
)
from hsr4hci.coordinates import get_center, polar2cartesian
from hsr4hci.data import load_metadata, load_psf_template
from hsr4hci.evaluation import compute_optimized_snr
from hsr4hci.fits import read_fits
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
    print('\nRUN HYPOTHESIS-BASED HALF-SIBLING REGRESSION\n', flush=True)

    # -------------------------------------------------------------------------
    # Load experiment configuration and data; parse command line arguments
    # -------------------------------------------------------------------------

    # Set up parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-dir',
        type=str,
        required=True,
        metavar='PATH',
        help='(Absolute) path to experiment directory.',
    )
    args = parser.parse_args()

    # Get experiment directory
    experiment_dir = Path(os.path.expanduser(args.experiment_dir))
    if not experiment_dir.exists():
        raise NotADirectoryError(f'{experiment_dir} does not exist!')

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    config = load_config(experiment_dir / 'config.json')
    if not 'hypothesis' in config.keys():
        raise RuntimeError('Experiment configuration contains no hypothesis!')
    print('Done!', flush=True)

    # Load the PSF template and metadata of the dataset
    print('Loading data set...', end=' ', flush=True)
    psf_template = load_psf_template(**config['dataset'])
    metadata = load_metadata(**config['dataset'])
    print('Done!', flush=True)

    # Load the signal estimate
    print('Loading signal estimate...', end=' ', flush=True)
    file_path = experiment_dir / 'results' / 'signal_estimate.fits'
    signal_estimate = np.asarray(read_fits(file_path))
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Define various useful shortcuts; activate unit conversions
    # -------------------------------------------------------------------------

    # Metadata of the data set
    pixscale = float(metadata['PIXSCALE'])
    lambda_over_d = float(metadata['LAMBDA_OVER_D'])
    dit_stack = float(metadata['DIT_STACK'])
    dit_psf_template = float(metadata['DIT_PSF_TEMPLATE'])
    nd_filter = float(metadata['ND_FILTER'])

    # Other shortcuts
    selected_keys = config['observing_conditions']['selected_keys']
    selection_mask_config = config['selection_mask']

    # Activate the unit conversions for this instrument
    set_units_for_instrument(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
        verbose=False,
    )

    # Quantities related to the size of the data set
    frame_size = signal_estimate.shape
    psf_size = (psf_template.shape[0], psf_template.shape[1])

    # -------------------------------------------------------------------------
    # STEP 1: Fit the PSF radius and measure the stellar flux
    # -------------------------------------------------------------------------

    # Fit the FWHM of the PSF template
    psf_fwhm = get_psf_fwhm(psf_template)
    print(f'\nPSF FWHM: {psf_fwhm:.2f} pixel\n')

    # Compute the scaling factor for measuring the stellar flux:
    # We need to account for the different integration times between the PSF
    # template and the stack, as well as the fact that for the PSF template,
    # some observations use a neutral density filter that filters out most of
    # the light (for NACO, the transmissibility is usually around 2%).
    scaling_factor = 1 / nd_filter * dit_stack / dit_psf_template

    # Measure stellar flux from the unsaturated PSF template
    stellar_flux = float(
        get_aperture_flux(
            frame=psf_template * scaling_factor,
            position=get_center(psf_size),
            aperture_radius=Quantity(psf_fwhm / 2, 'pixel'),
        )
    )

    # -------------------------------------------------------------------------
    # STEP 2: Compute optimized signal-to-noise ratio
    # -------------------------------------------------------------------------

    # Keep track of all results which will be saved to a TSV file
    results = {}

    # Loop over all planets in the data set
    print('\n\nSIGNAL-TO-NOISE RATIO:\n')
    for name, parameters in config['hypothesis'].items():

        # Compute the expected planet position in Cartesian coordinates
        planet_position = polar2cartesian(
            separation=Quantity(*parameters['separation']),
            angle=Quantity(*parameters['position_angle']),
            frame_size=frame_size,
        )

        # Compute the optimized SNR (actually, we optimize the flux in the
        # signal aperture, which is also why we use this position below)
        results[name] = compute_optimized_snr(
            frame=signal_estimate,
            position=planet_position,
            aperture_radius=Quantity(psf_fwhm / 2, 'pixel'),
            ignore_neighbors=1,
            max_distance=2,
            grid_size=40,
        )

        # Unpack results dict
        signal = results[name]['signal']
        noise = results[name]['noise']
        snr = results[name]['snr']
        fpf = results[name]['fpf']

        print(
            f'  {name}: SNR = {snr:.2f} (FPF = {fpf:.2e} | '
            f'signal = {signal:.2e}) | noise = {noise:.2e})'
        )

    # -------------------------------------------------------------------------
    # STEP 4: Run photometry
    # -------------------------------------------------------------------------

    # Loop over all planets in the data set
    print('\n\nCONTRAST:\n')
    for name in results.keys():

        # Get the optimized position (i.e., position with maximum flux)
        optimized_position = results[name]['new_position']

        # Determine the reference aperture positions and measure the fluxes
        reference_positions, _ = get_reference_aperture_positions(
            frame_size=frame_size,
            position=optimized_position,
            aperture_radius=Quantity(psf_fwhm / 2, 'pixel'),
            ignore_neighbors=1,
        )
        reference_flux = get_aperture_flux(
            frame=signal_estimate,
            position=reference_positions,
            aperture_radius=Quantity(psf_fwhm / 2, 'pixel'),
        )

        # Measure the planet flux and subtract the mean reference flux (this
        # is our estimate for the leftover noise at the planet position)
        raw_planet_flux = float(
            get_aperture_flux(
                frame=signal_estimate,
                position=optimized_position,
                aperture_radius=Quantity(psf_fwhm / 2, 'pixel'),
            )
        )
        mean_reference_flux = float(np.mean(reference_flux))
        planet_flux = raw_planet_flux - mean_reference_flux

        # Compute brightness ratio of the star and the planet in magnitudes;
        # we call this the observed contrast
        flux_ratio_observed = planet_flux / stellar_flux
        contrast_observed = flux_ratio_to_magnitudes(flux_ratio_observed)

        # Convert the expected contrast (from the literature) to a flux ratio
        contrast_expected = config['hypothesis'][name]['contrast']
        flux_ratio_expected = magnitude_to_flux_ratio(contrast_expected)

        # Compute "throughput" (ratio of our magnitude vs. true magnitude)
        throughput = flux_ratio_observed / flux_ratio_expected

        # Store results
        results[name]['raw_planet_flux'] = raw_planet_flux
        results[name]['mean_reference_flux'] = mean_reference_flux
        results[name]['planet_flux'] = planet_flux
        results[name]['flux_ratio_observed'] = flux_ratio_observed
        results[name]['contrast_observed'] = contrast_observed
        results[name]['flux_ratio_expected'] = flux_ratio_expected
        results[name]['contrast_expected'] = contrast_expected
        results[name]['throughput'] = throughput

        print(
            f'  {name}: {contrast_observed:.2f} mag '
            f'(throughput: {throughput:.3f})'
        )

    # -------------------------------------------------------------------------
    # STEP 5: Convert results to data frame and save them as a TSV
    # -------------------------------------------------------------------------

    # Convert to data frame
    results_df = pd.DataFrame(results)

    # Store results as TSV file
    print('\nSaving results to TSV...', end=' ', flush=True)
    file_path = experiment_dir / 'results' / 'results.tsv'
    results_df.to_csv(file_path, sep='\t')
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
