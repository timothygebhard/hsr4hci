"""
This script runs a PCA-based PSF subtraction on the data sets created by
the 01_create_data_sets.py script, and computes the respective signal
estimates (on which we can then compute the baseline SNR).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from copy import deepcopy
from pathlib import Path

import json
import os
import time

from astropy import units

import numpy as np

from hsr4hci.utils.argparsing import get_base_directory
from hsr4hci.utils.data import load_data
from hsr4hci.utils.fits import save_fits
from hsr4hci.utils.masking import get_roi_mask
from hsr4hci.utils.pca import get_pca_signal_estimates
from hsr4hci.utils.units import convert_to_quantity, set_units_for_instrument


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nRUN PCA AND COMPUTE SIGNAL ESTIMATES\n', flush=True)

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
    config['metadata']['PIXSCALE'] = \
        units.Quantity(config['metadata']['PIXSCALE'], 'arcsec / pixel')
    config['metadata']['LAMBDA_OVER_D'] = \
        units.Quantity(config['metadata']['LAMBDA_OVER_D'], 'arcsec')

    # Use this to set up the instrument-specific conversion factors. We need
    # this here to that we can parse "lambda_over_d" as a unit in the config.
    set_units_for_instrument(pixscale=config['metadata']['PIXSCALE'],
                             lambda_over_d=config['metadata']['LAMBDA_OVER_D'])

    # Convert the relevant entries of the config to astropy.units.Quantity
    for key_tuple in [('roi', 'inner_radius'),
                      ('roi', 'outer_radius')]:
        config = convert_to_quantity(config, key_tuple)

    # -------------------------------------------------------------------------
    # Define shortcuts to various parts of the config
    # -------------------------------------------------------------------------

    # Get stacking factors
    stacking_factors = config['stacking_factors']

    # Construct the pca_numbers for the PcaPsfSubtractionModule
    min_n_components = config['pca']['min_n_components']
    max_n_components = config['pca']['max_n_components']
    pca_numbers = list(range(min_n_components, max_n_components + 1))

    # Construct a ROI mask
    roi_mask = get_roi_mask(mask_size=config['frame_size'],
                            inner_radius=config['roi']['inner_radius'],
                            outer_radius=config['roi']['outer_radius'])

    # -------------------------------------------------------------------------
    # Run PCA
    # -------------------------------------------------------------------------

    for stacking_factor in stacking_factors:

        print(f'Running for stacking factor {stacking_factor}:')
        print(80 * '-')

        # ---------------------------------------------------------------------
        # Create a directory in which we store the results
        # ---------------------------------------------------------------------

        result_dir = os.path.join(base_dir, 'pca_baselines',
                                  f'stacked_{stacking_factor}')
        Path(result_dir).mkdir(exist_ok=True, parents=True)

        # ---------------------------------------------------------------------
        # Load the data and run our custom PCA to compute signal estimates
        # ---------------------------------------------------------------------

        # Load the input stack and the parallactic angles
        file_path = os.path.join(base_dir, 'processed',
                                 f'stacked_{stacking_factor}.hdf')
        stack, parang, _ = load_data(file_path=file_path)

        # Apply the ROI mask to the input stack (we have to use 0 instead of
        # NaN here, because the PCA cannot deal with NaNs)
        stack[:, ~roi_mask] = 0

        # Compute the signal estimates and the principal components
        signal_estimates, principal_components = \
            get_pca_signal_estimates(stack=stack,
                                     parang=parang,
                                     pca_numbers=pca_numbers,
                                     return_components=True,
                                     verbose=True)

        # Apply the ROI mask to the signal estimates
        signal_estimates[:, ~roi_mask] = np.nan

        # ---------------------------------------------------------------------
        # Save the signal estimates and principal components as FITS files
        # ---------------------------------------------------------------------

        # Save signal estimates as FITS file
        print('Saving signal estimates to FITS...', end=' ', flush=True)
        file_path = os.path.join(result_dir, 'signal_estimates.fits')
        save_fits(array=signal_estimates, file_path=file_path)
        print('Done!', flush=True)

        # Save principal components as FITS file
        print('Saving principal components to FITS...', end=' ', flush=True)
        file_path = os.path.join(result_dir, 'principal_components.fits')
        save_fits(array=principal_components, file_path=file_path)
        print('Done!', flush=True)

        print(80 * '-', '\n', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'This took {time.time() - script_start:.1f} seconds!\n')
