"""
Inject fake planet and run a PCA-based post-processing pipeline.
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
from hsr4hci.forward_modeling import add_fake_planet
from hsr4hci.pca import get_pca_signal_estimates
from hsr4hci.psf import get_psf_fwhm
from hsr4hci.positions import get_injection_position
from hsr4hci.units import InstrumentUnitsContext


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nINJECT FAKE PLANET AND RUN PCA\n', flush=True)

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
    # By default, the stack is already loaded *without* the planet
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

    # Define the unit conversion context for this data set
    instrument_units_context = InstrumentUnitsContext(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
    )

    # Fit the FWHM of the PSF (in pixels)
    psf_fwhm = get_psf_fwhm(psf_template)

    # -------------------------------------------------------------------------
    # Inject a fake planet into the stack
    # -------------------------------------------------------------------------

    # Get injection parameters
    contrast = config['injection']['contrast']
    separation = config['injection']['separation']
    azimuthal_position = config['injection']['azimuthal_position']

    # If any parameter is None, skip the injection...
    if contrast is None or separation is None or azimuthal_position is None:
        print('Skipping injection of a fake planet!', flush=True)

    # ... otherwise, add a fake planet with given parameters to the stack
    else:

        # Convert separation from units of FWHM to pixel
        separation *= psf_fwhm

        # Compute position at which to inject the fake planet
        print('Computing injection position...', end=' ', flush=True)
        injection_position = get_injection_position(
            separation=Quantity(separation, 'pixel'),
            azimuthal_position=azimuthal_position,
        )
        print(
            f'Done! (separation = {separation:.1f} pixel, '
            f'azimuthal_position = {azimuthal_position})',
            flush=True,
        )

        # Inject the fake planet at the injection_position
        print('Injecting fake planet...', end=' ', flush=True)
        stack = np.asarray(
            add_fake_planet(
                stack=stack,
                parang=parang,
                psf_template=psf_template,
                polar_position=injection_position,
                magnitude=contrast,
                extra_scaling=(1.0 / float(metadata['ND_FILTER'])),
                dit_stack=float(metadata['DIT_STACK']),
                dit_psf_template=float(metadata['DIT_PSF_TEMPLATE']),
                return_planet_positions=False,
                interpolation='bilinear',
            )
        )
        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Run PCA to get signal estimate and save as a FITS files
    # -------------------------------------------------------------------------

    # Run PCA (for a fixed number of principal components)
    print('Running PCA...', end=' ', flush=True)
    with instrument_units_context:
        signal_estimate = get_pca_signal_estimates(
            stack=stack,
            parang=parang,
            n_components=int(config['n_components']),
            roi_mask=None,
            return_components=False,
        ).squeeze()
    print('Done!', flush=True)

    # Save signal estimate to FITS
    print('Saving signal estimate to FITS...', end=' ', flush=True)
    file_path = results_dir / 'signal_estimate.fits'
    save_fits(array=signal_estimate, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
