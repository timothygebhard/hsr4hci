"""
Run a PCA-based post-processing pipeline.
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

from hsr4hci.config import load_config
from hsr4hci.data import load_dataset
from hsr4hci.fits import save_fits
from hsr4hci.masking import get_roi_mask
from hsr4hci.pca import get_pca_signal_estimates
from hsr4hci.units import InstrumentUnitsContext


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nRUN PCA-BASED POST-PROCESSING\n', flush=True)

    # -------------------------------------------------------------------------
    # Set up parser to get command line arguments
    # -------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-dir',
        type=str,
        required=True,
        metavar='PATH',
        help='Path to experiment directory.',
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load experiment configuration and data
    # -------------------------------------------------------------------------

    # Get experiment directory
    experiment_dir = Path(os.path.expanduser(args.experiment_dir))
    if not experiment_dir.exists():
        raise NotADirectoryError(f'{experiment_dir} does not exist!')

    # Get path to results directory
    results_dir = experiment_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    config = load_config(experiment_dir / 'config.json')
    print('Done!', flush=True)

    # Load frames, parallactic angles, etc. from HDF file
    print('Loading data set...', end=' ', flush=True)
    stack, parang, psf_template, observing_conditions, metadata = load_dataset(
        **config['dataset']
    )
    print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Define various useful shortcuts; activate unit conversions
    # -------------------------------------------------------------------------

    # Quantities related to the size of the data set
    n_frames, x_size, y_size = stack.shape
    frame_size = (x_size, y_size)

    # Metadata of the data set
    pixscale = float(metadata['PIXSCALE'])
    lambda_over_d = float(metadata['LAMBDA_OVER_D'])

    # Define a unit conversion context
    instrument_unit_context = InstrumentUnitsContext(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
    )

    # Get minimum/maximum number of PCs, as well as the default value to
    # choose "the" one single signal estimate
    min_n = int(config['pca']['min_n'])
    max_n = int(config['pca']['max_n'])
    default_n = int(config['pca']['default_n'])
    default_idx = default_n - min_n
    n_components = [int(_) for _ in np.arange(min_n, max_n)]

    # Compute ROI mask
    with instrument_unit_context:
        roi_mask = get_roi_mask(
            mask_size=frame_size,
            inner_radius=Quantity(*config['roi_mask']['inner_radius']),
            outer_radius=Quantity(*config['roi_mask']['outer_radius']),
        )

    # -------------------------------------------------------------------------
    # Run PCA to get signal estimates and principal components
    # -------------------------------------------------------------------------

    # Run PCA to get signal estimates and principal components
    print('Computing PCA-based signal estimates...', end=' ', flush=True)
    signal_estimates, principal_components = get_pca_signal_estimates(
        stack=stack,
        parang=parang,
        n_components=n_components,
        return_components=True,
        roi_mask=None,
    )
    print('Done!\n', flush=True)

    # Apply ROI mask *after* PCA
    signal_estimates[:, ~roi_mask] = np.nan

    # Select "the" signal estimate
    signal_estimate = signal_estimates[default_idx]

    # -------------------------------------------------------------------------
    # Save results to FITS files
    # -------------------------------------------------------------------------

    print('Saving results...', end=' ', flush=True)
    file_path = results_dir / 'principal_components.fits'
    save_fits(array=principal_components, file_path=file_path)
    file_path = results_dir / 'signal_estimates.fits'
    save_fits(array=signal_estimates, file_path=file_path)
    file_path = results_dir / 'signal_estimate.fits'
    save_fits(array=signal_estimate, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
