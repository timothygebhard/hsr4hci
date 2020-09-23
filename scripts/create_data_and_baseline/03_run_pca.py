"""
This script runs a PCA-based PSF subtraction on the data sets created by
the 02_create_data_sets.py script, and computes the respective signal
estimates (on which we can then compute the baseline SNR).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import json
import time

from astropy.units import Quantity

import h5py
import numpy as np

from hsr4hci.utils.argparsing import get_base_directory
from hsr4hci.utils.fits import save_fits
from hsr4hci.utils.masking import get_roi_mask
from hsr4hci.utils.pca import get_pca_signal_estimates
from hsr4hci.utils.units import set_units_for_instrument


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

    # Read in the config file and parse it
    file_path = base_dir / 'config.json'
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)

    # -------------------------------------------------------------------------
    # Define shortcuts and set up unit conversions
    # -------------------------------------------------------------------------

    # Shortcuts to PIXSCALE and LAMBDA_OVER_D
    metadata = config['metadata']
    pixscale = metadata['PIXSCALE']
    lambda_over_d = metadata['LAMBDA_OVER_D']

    # Use this to set up the instrument-specific conversion factors. We need
    # this here to that we can parse "lambda_over_d" as a unit in the config.
    set_units_for_instrument(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
    )

    # Other shortcuts to elements in the config
    stacking_factors = config['stacking_factors']
    min_n_components = config['pca']['min_n_components']
    max_n_components = config['pca']['max_n_components']
    pca_numbers = list(range(min_n_components, max_n_components + 1))

    # -------------------------------------------------------------------------
    # Construct a ROI mask
    # -------------------------------------------------------------------------

    roi_mask = get_roi_mask(
        mask_size=config['frame_size'],
        inner_radius=Quantity(*config['roi']['inner_radius']),
        outer_radius=Quantity(*config['roi']['outer_radius']),
    )

    # -------------------------------------------------------------------------
    # Run PCA
    # -------------------------------------------------------------------------

    for stacking_factor in stacking_factors:

        print(f'Running for stacking factor {stacking_factor}:')
        print(80 * '-')

        # ---------------------------------------------------------------------
        # Create a directory in which we store the results
        # ---------------------------------------------------------------------

        result_dir = base_dir / 'pca_baselines' / f'stacked_{stacking_factor}'
        result_dir.mkdir(exist_ok=True, parents=True)

        # ---------------------------------------------------------------------
        # Load the data and run our custom PCA to compute signal estimates
        # ---------------------------------------------------------------------

        # Load the input stack and the parallactic angles
        file_path = base_dir / 'processed' / f'stacked__{stacking_factor}.hdf'
        with h5py.File(file_path, 'r') as hdf_file:
            stack = np.array(hdf_file['stack'])
            parang = np.array(hdf_file['parang'])

        # Apply the ROI mask to the input stack (we have to use 0 instead of
        # NaN here, because the PCA cannot deal with NaNs)
        stack[:, ~roi_mask] = 0

        # Compute the signal estimates and the principal components
        signal_estimates, principal_components = get_pca_signal_estimates(
            stack=stack,
            parang=parang,
            pca_numbers=pca_numbers,
            return_components=True,
            verbose=True,
        )

        # Apply the ROI mask to the signal estimates
        signal_estimates[:, ~roi_mask] = np.nan

        # ---------------------------------------------------------------------
        # Save the signal estimates and principal components as FITS files
        # ---------------------------------------------------------------------

        # Save signal estimates as FITS file
        print('Saving signal estimates to FITS...', end=' ', flush=True)
        file_path = result_dir / 'signal_estimates.fits'
        save_fits(array=signal_estimates, file_path=file_path)
        print('Done!', flush=True)

        # Save principal components as FITS file
        print('Saving principal components to FITS...', end=' ', flush=True)
        file_path = result_dir / 'principal_components.fits'
        save_fits(array=principal_components, file_path=file_path)
        print('Done!', flush=True)

        print(80 * '-', '\n', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'This took {time.time() - script_start:.1f} seconds!\n')
