"""
This script creates a simple median ADI baseline for each data set.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

import json
import os
import time

from astropy import units

import pandas as pd

from hsr4hci.utils.argparsing import get_base_directory
from hsr4hci.utils.data import load_data
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

    # Construct (expected) path to config.json
    file_path = os.path.join(base_dir, 'config.json')

    # Read in the config file and parse it
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)

    # Get a copy of the pixscale without units
    pixscale = deepcopy(config['metadata']['PIXSCALE'])

    # Now, apply unit conversions to astropy.units:
    # First, convert pixscale and lambda_over_d to astropy.units.Quantity. This
    # is a bit cumbersome, because in the meta data, we cannot use the usual
    # convention to specify units, as the meta data are also written to the HDF
    # file. Hence, we must hard-code the unit conventions here.
    config['metadata']['PIXSCALE'] = units.Quantity(
        config['metadata']['PIXSCALE'], 'arcsec / pixel'
    )
    config['metadata']['LAMBDA_OVER_D'] = units.Quantity(
        config['metadata']['LAMBDA_OVER_D'], 'arcsec'
    )

    # Use this to set up the instrument-specific conversion factors. We need
    # this here to that we can parse "lambda_over_d" as a unit in the config.
    set_units_for_instrument(
        pixscale=config['metadata']['PIXSCALE'],
        lambda_over_d=config['metadata']['LAMBDA_OVER_D'],
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

        result_dir = os.path.join(
            base_dir, 'median_adi_baselines', f'stacked_{stacking_factor}'
        )
        Path(result_dir).mkdir(exist_ok=True, parents=True)

        # ---------------------------------------------------------------------
        # Load the data and run median ADI to get the signal estimate
        # ---------------------------------------------------------------------

        # Load the input stack and the parallactic angles
        file_path = os.path.join(
            base_dir, 'processed', f'stacked_{stacking_factor}.hdf'
        )
        stack, parang, _, __, ___ = load_data(file_path=file_path)

        # Compute the signal estimates and the principal components
        signal_estimate = derotate_combine(
            stack=stack, parang=parang, mask=roi_mask, subtract='median'
        )

        # ---------------------------------------------------------------------
        # Save the signal estimates as FITS files
        # ---------------------------------------------------------------------

        # Save signal estimates as FITS file
        print('Saving signal estimates to FITS...', end=' ', flush=True)
        file_path = os.path.join(result_dir, 'signal_estimate.fits')
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
        snr_options = config['evaluation']['snr_options']
        aperture_radius = to_pixel(snr_options['aperture_radius'])
        max_distance = to_pixel(snr_options['max_distance'])
        method = snr_options['method']
        target = snr_options['target']
        time_limit = snr_options['time_limit']

        for planet_key, planet_position in planet_positions.items():

            # Construct full name of the planet (target star + letter)
            planet_name = f'{config["metadata"]["TARGET_STAR"]} {planet_key}'

            print(f'Running SNR computation for {planet_name}:', flush=True)

            # Actually compute the SNR and other quantities
            result = compute_optimized_snr(
                frame=signal_estimate,
                position=planet_position,
                aperture_radius=aperture_radius,
                ignore_neighbors=ignore_neighbors[planet_key],
                target=target,
                method=method,
                max_distance=max_distance,
                time_limit=time_limit,
            )

            # Save result for each planet individually as a CSV file
            file_path = os.path.join(result_dir, f'{planet_name}.csv')
            pd.DataFrame(result).to_csv(file_path)

            print('Done!', flush=True)
            print(80 * '-', '\n', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'This took {time.time() - script_start:.1f} seconds!\n')
