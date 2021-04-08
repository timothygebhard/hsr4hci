"""
Cut patches from each residual frame to estimate planetary PSF.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import os
import time

from astropy.units import Quantity

import h5py
import numpy as np

from hsr4hci.config import load_config
from hsr4hci.coordinates import get_center
from hsr4hci.data import (
    load_metadata,
    load_parang,
    load_planets,
)
from hsr4hci.fits import read_fits, save_fits
from hsr4hci.general import crop_around_position_with_interpolation
from hsr4hci.masking import get_positions_from_mask
from hsr4hci.units import set_units_for_instrument


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nGET PSF FROM RESIDUAL PATCHES\n', flush=True)

    # -------------------------------------------------------------------------
    # Set up parser to get command line arguments
    # -------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-dir',
        type=str,
        required=True,
        metavar='PATH',
        help='(Absolute) path to experiment directory.',
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        metavar='SIZE',
        default=25,
        help='Width/height (in pixels) of the patches to be cropped.',
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load experiment configuration and data; parse command line arguments
    # -------------------------------------------------------------------------

    # Get experiment directory
    experiment_dir = Path(os.path.expanduser(args.experiment_dir))
    if not experiment_dir.exists():
        raise NotADirectoryError(f'{experiment_dir} does not exist!')
    results_dir = experiment_dir / 'results'

    # Get patch size from command line arguments
    patch_size = int(args.patch_size)

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    config = load_config(experiment_dir / 'config.json')
    print('Done!', flush=True)

    # Load parallactic angles, PSF template, metadata and planets
    parang = load_parang(**config['dataset'])
    metadata = load_metadata(**config['dataset'])
    planets = load_planets(**config['dataset'])

    # Activate the unit conversions for this instrument
    set_units_for_instrument(
        pixscale=Quantity(metadata['PIXSCALE'], 'arcsec / pixel'),
        lambda_over_d=Quantity(metadata['LAMBDA_OVER_D'], 'arcsec'),
        verbose=False,
    )

    # -------------------------------------------------------------------------
    # Load or construct residuals
    # -------------------------------------------------------------------------

    # If there already exists a residuals.fits file, we load it directly...
    if (results_dir / 'residuals.fits').exists():

        print('\nLoading selection mask from FITS...', end=' ', flush=True)
        file_path = results_dir / 'residuals.fits'
        residuals = np.asarray(read_fits(file_path))
        print('Done!', flush=True)

    # Otherwise (e.g., for HSR), we need to (re-)construct the residuals
    else:

        # Load selection mask from FITS
        print('\nLoading selection mask from FITS...', end=' ', flush=True)
        file_path = results_dir / 'selection_mask.fits'
        selection_mask = np.asarray(read_fits(file_path))
        print('Done!', flush=True)

        # Load hypotheses from FITS
        print('Loading hypotheses from FITS...', end=' ', flush=True)
        file_path = experiment_dir / 'hypotheses' / 'hypotheses.fits'
        hypotheses = np.asarray(read_fits(file_path))
        print('Done!', flush=True)

        # Load residuals from HDF (based on hypotheses and selection mask)
        print('Constructing residuals...', end=' ', flush=True)
        file_path = experiment_dir / 'hdf' / 'results.hdf'
        with h5py.File(file_path, 'r') as hdf_file:

            # Initialize everything to the default residuals
            residuals = np.array(hdf_file['default']['residuals'])

            # For the pixels where the selection mask is 1, choose the
            # residuals based on the corresponding hypothesis
            for (x, y) in get_positions_from_mask(selection_mask):
                signal_time = str(int(hypotheses[x, y]))
                residuals[:, x, y] = np.array(
                    hdf_file[signal_time]['residuals'][:, x, y]
                )
        print('Done!', flush=True)

    # Define shortcuts
    n_frames, x_size, y_size = residuals.shape
    frame_size = (x_size, y_size)

    # -------------------------------------------------------------------------
    # Compute expected stack based on hypothesis
    # -------------------------------------------------------------------------

    print('Computing hypothesized stack(s)...', end=' ', flush=True)

    # Initialize the hypothesized stack and mask of affected pixels
    planet_positions = {}

    # Loop over potentially multiple planet hypotheses and add them
    for name, parameters in planets.items():

        # Compute planet positions in polar coordinates
        rho = Quantity(parameters['separation'], 'arcsec').to('pixel').value
        phi = np.radians(parameters['position_angle'] + 90 - parang)

        # Compute the offset from the center in Cartesian coordinates
        center = get_center(frame_size)
        x_shift = rho * np.cos(phi)
        y_shift = rho * np.sin(phi)

        # Compute and store the planet positions in Cartesian coordinates
        # Note: these coordinates follow the *numpy* convention
        planet_positions[name] = np.column_stack(
            (y_shift + center[1], x_shift + center[0])
        )

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Crop PSF patches from residual frames
    # -------------------------------------------------------------------------

    # Create a directory where we store the results
    psf_dir = experiment_dir / 'psf'
    psf_dir.mkdir(exist_ok=True)

    # Loop over planets
    for name, parameters in planets.items():

        print(f'\nCropping patches for planet {name}...', end=' ', flush=True)

        # Store patches for this planet
        patches = []

        # Loop over residual frames and crop the
        for frame, position in zip(residuals, planet_positions[name]):
            try:
                patch = crop_around_position_with_interpolation(
                    array=np.nan_to_num(frame),
                    position=position,
                    size=(patch_size, patch_size),
                )
                patches.append(patch)
            except ValueError as error:
                if 'One of the requested xi is out of bounds' in str(error):
                    print('\n\nERROR: Patch size too large!\n')
                raise

        # Convert the list of patches to a numpy array
        patches = np.array(patches)

        # Store the patches as a FITS file
        print('Saving patches to FITS...', end=' ', flush=True)
        file_path = psf_dir / f'psf_patches__{name}.fits'
        save_fits(patches, file_path)
        print('Done!', flush=True)

        # Store the median patch (= PSF estimate) as a FITS file
        print('Saving PSF estimate to FITS...', end=' ', flush=True)
        file_path = psf_dir / f'psf_estimate__{name}.fits'
        save_fits(np.median(patches, axis=0), file_path)
        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
