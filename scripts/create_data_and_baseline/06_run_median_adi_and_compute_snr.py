"""
This script creates a simple median ADI baseline for each data set.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Dict, Tuple

import csv
import json
import time

from astropy.units import Quantity

import h5py
import numpy as np

from hsr4hci.utils.argparsing import get_base_directory
from hsr4hci.utils.derotating import derotate_combine
from hsr4hci.utils.evaluation import compute_optimized_snr
from hsr4hci.utils.fits import save_fits
from hsr4hci.utils.masking import get_roi_mask
from hsr4hci.utils.units import (
    convert_to_quantity,
    to_pixel,
    set_units_for_instrument,
)


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nRUN MEDIAN ADI AND COMPUTE SIGNAL ESTIMATES\n', flush=True)

    # -------------------------------------------------------------------------
    # Parse command line arguments and load config.json
    # -------------------------------------------------------------------------

    # Get base_directory from command line arguments
    base_dir = get_base_directory()

    # Read in the config file and parse it
    file_path = base_dir / 'config.json'
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)

    # Define shortcuts to values in config
    metadata = config['metadata']
    pixscale = metadata['PIXSCALE']
    lambda_over_d = metadata['LAMBDA_OVER_D']

    # Use this to set up the instrument-specific conversion factors. We need
    # this here to that we can parse "lambda_over_d" as a unit in the config.
    set_units_for_instrument(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
    )

    # Convert the relevant entries of the config to astropy.units.Quantity
    for key_tuple in [
        ('roi', 'inner_radius'),
        ('roi', 'outer_radius'),
        ('evaluation', 'snr_options', 'aperture_radius'),
        ('evaluation', 'snr_options', 'max_distance'),
    ]:
        config = convert_to_quantity(config, key_tuple)

    # -------------------------------------------------------------------------
    # Define shortcuts to various parts of the config
    # -------------------------------------------------------------------------

    # Get stacking factors
    stacking_factors = config['stacking_factors']

    # Construct a ROI mask
    roi_mask = get_roi_mask(
        mask_size=config['frame_size'],
        inner_radius=config['roi']['inner_radius'],
        outer_radius=config['roi']['outer_radius'],
    )

    # -------------------------------------------------------------------------
    # Run median ADI and compute SNR
    # -------------------------------------------------------------------------

    for stacking_factor in stacking_factors:

        print(f'Running for stacking factor {stacking_factor}:')
        print(80 * '-')

        # ---------------------------------------------------------------------
        # Create a directory in which we store the results
        # ---------------------------------------------------------------------

        result_dir = (
            base_dir / 'median_adi_baselines' / f'stacked__{stacking_factor}'
        )
        result_dir.mkdir(exist_ok=True, parents=True)

        # ---------------------------------------------------------------------
        # Load the data and run median ADI to get the signal estimate
        # ---------------------------------------------------------------------

        # Load the input stack and the parallactic angles
        file_path = base_dir / 'processed' / f'stacked__{stacking_factor}.hdf'
        with h5py.File(file_path, 'r') as hdf_file:
            stack = np.array(hdf_file['stack'])
            parang = np.array(hdf_file['parang'])

        # Compute the signal estimates and the principal components
        print('Computing median ADI signal estimate...', end=' ', flush=True)
        signal_estimate = derotate_combine(
            stack=stack, parang=parang, mask=~roi_mask, subtract='median'
        )
        print('Done!', flush=True)

        # ---------------------------------------------------------------------
        # Save the signal estimates as FITS files
        # ---------------------------------------------------------------------

        # Save signal estimates as FITS file
        print('Saving signal estimate to FITS...', end=' ', flush=True)
        file_path = result_dir / 'signal_estimate.fits'
        save_fits(array=signal_estimate, file_path=file_path)
        print('Done!', flush=True)

        # ---------------------------------------------------------------------
        # Compute SNR and related quantities
        # ---------------------------------------------------------------------

        # Define shortcuts for planet positions
        planet_positions: Dict[str, Tuple[float, float]] = dict()
        ignore_neighbors: Dict[str, int] = dict()
        for planet_key, options in config['evaluation']['planets'].items():
            planet_positions[planet_key] = (
                float(options['position'][0]),
                float(options['position'][1]),
            )
            ignore_neighbors[planet_key] = int(options['ignore_neighbors'])

        # Define shortcuts for SNR options
        target_star = config["metadata"]["TARGET_STAR"]
        snr_options = config['evaluation']['snr_options']
        aperture_radius = to_pixel(snr_options['aperture_radius'])
        max_distance = to_pixel(snr_options['max_distance'])
        method = snr_options['method']
        target = snr_options['target']
        grid_size = snr_options['grid_size']
        time_limit = snr_options['time_limit']

        for planet_key, planet_position in planet_positions.items():

            # Construct full name of the planet (target star + letter)
            planet_name = f'{target_star} {planet_key}'

            print(f'Computing SNR for {planet_name}...', end=' ', flush=True)

            # Actually compute the SNR and other quantities
            result = compute_optimized_snr(
                frame=signal_estimate,
                position=planet_position,
                aperture_radius=aperture_radius,
                ignore_neighbors=ignore_neighbors[planet_key],
                target=target,
                method=method,
                max_distance=max_distance,
                grid_size=grid_size,
                time_limit=time_limit,
            )

            # Save result for each planet individually as a CSV file
            file_path = result_dir / f'{planet_key}.csv'
            with open(file_path, 'w') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerows(result.items())

            print('Done!', flush=True)

        print(80 * '-', '\n', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'This took {time.time() - script_start:.1f} seconds!\n')
