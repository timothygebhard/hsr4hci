"""
Merge partial residual FITS files and compute signal estimate.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from itertools import product
from pathlib import Path

import argparse
import os
import time

from astropy.units import Quantity
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

import numpy as np

from hsr4hci.config import load_config
from hsr4hci.data import load_parang, load_metadata, load_psf_template
from hsr4hci.derotating import derotate_combine
from hsr4hci.fits import read_fits, save_fits
from hsr4hci.forward_modeling import add_fake_planet
from hsr4hci.masking import get_roi_mask, get_partial_roi_mask
from hsr4hci.merging import get_list_of_fits_file_paths
from hsr4hci.units import InstrumentUnitsContext


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nMERGE RESIDUAL FILES AND COMPUTE SIGNAL ESTIMATE\n', flush=True)

    # -------------------------------------------------------------------------
    # Set up parser to get command line arguments; get experiment directory
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
    print('Done!', flush=True)

    # Load metadata, parallactic angles and PSF template
    metadata = load_metadata(**config['dataset'])
    parang = load_parang(**config['dataset'])
    psf_template = load_psf_template(**config['dataset'])

    # Define the unit conversion context for this data set
    instrument_unit_context = InstrumentUnitsContext(
        pixscale=Quantity(metadata['PIXSCALE'], 'arcsec / pixel'),
        lambda_over_d=Quantity(metadata['LAMBDA_OVER_D'], 'arcsec'),
    )

    # -------------------------------------------------------------------------
    # Collect and merge partial residual files
    # -------------------------------------------------------------------------

    # Define shortcuts
    n_frames = len(parang)
    x_size = config['dataset']['frame_size'][0]
    y_size = config['dataset']['frame_size'][1]
    frame_size = (x_size, y_size)

    # Construct the mask for the region of interest (ROI)
    with instrument_unit_context:
        roi_mask = get_roi_mask(
            mask_size=frame_size,
            inner_radius=Quantity(*config['roi_mask']['inner_radius']),
            outer_radius=Quantity(*config['roi_mask']['outer_radius']),
        )

    # Define directory that contains partial residuals
    partial_dir = experiment_dir / 'fits' / 'partial'

    # Collect paths to FITS files that we need to merge
    fits_file_paths = get_list_of_fits_file_paths(
        fits_dir=partial_dir, prefix='residuals'
    )

    # Initialize an array for the full residual stack
    residual_stack = np.full((n_frames, x_size, y_size), np.nan)

    # Loop over the FITS files that we need to merge
    print('\nMerging partial residuals:', flush=True)
    for file_path in tqdm(fits_file_paths, ncols=80):

        # Parse the file name to get the roi_split and n_roi_splits
        name = file_path.name.split('.')[0]
        roi_split = int(name.split('_')[1].split('-')[0]) - 1
        n_roi_splits = int(name.split('_')[1].split('-')[1])

        # Construct partial ROI mask that matches this file
        partial_roi_mask = get_partial_roi_mask(
            roi_mask=roi_mask,
            roi_split=roi_split,
            n_roi_splits=n_roi_splits,
        )

        # Read in the array with the partial residuals and place it at the
        # positions given by the partial ROI mask
        array = read_fits(file_path, return_header=False)
        residual_stack[:, partial_roi_mask] = array

    # Finally, save the merged FITS files
    print('\nSaving merged residuals to FITS...', end=' ', flush=True)
    file_path = experiment_dir / 'fits' / 'residuals.fits'
    save_fits(array=residual_stack, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Derotate frames to compute the signal estimate
    # -------------------------------------------------------------------------

    # Ensure that the results directory exists
    results_dir = experiment_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    # Compute signal estimate
    print('Computing signal estimate...', end=' ', flush=True)
    signal_estimate = derotate_combine(
        stack=residual_stack, parang=parang, mask=~roi_mask
    )
    print('Done!', flush=True)

    print('Saving signal estimate to FITS...', end=' ', flush=True)
    file_path = results_dir / 'signal_estimate.fits'
    save_fits(array=signal_estimate, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Compute the hypothesized stack (for sanity check)
    # -------------------------------------------------------------------------

    print('\nComputing hypothesized stack...', end=' ', flush=True)

    # Make sure the PSF template is correctly normalized
    psf_template /= np.max(psf_template)

    # Initialize the hypothesized stack
    hypothesized_stack = np.zeros((n_frames, frame_size[0], frame_size[1]))

    # Loop over planet hypotheses and add fake planets to the stack
    with instrument_unit_context:
        for name, parameters in config['hypothesis'].items():
            hypothesized_stack = np.array(
                add_fake_planet(
                    stack=hypothesized_stack,
                    parang=parang,
                    psf_template=psf_template,
                    polar_position=(
                        Quantity(*parameters['separation']),
                        Quantity(*parameters['position_angle']),
                    ),
                    magnitude=0,
                    extra_scaling=1,
                    dit_stack=1,
                    dit_psf_template=1,
                    return_planet_positions=False,
                    interpolation='bilinear',
                )
            )
            hypothesized_stack /= np.max(hypothesized_stack)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Compute cosine similarities with hypothesized stack
    # -------------------------------------------------------------------------

    # Here, we compute the pixel-wise cosine similarity between the residual
    # stack and the hypothesized stack. If our hypothesis for the training
    # were correct, the resulting similarity frame should take on values close
    # to 1 everywhere on the trace of the planet.
    # If the trace is not very consistent, or contains artifacts, this might
    # indicate that our hypothesis was actually *incorrect*, and that the
    # "planet" that we might be seeing in the signal estimate needs to be
    # treated with caution as we might have just produced a false positive.

    print('\nComputing similarities...', end=' ', flush=True)
    similarities = np.full((x_size, y_size), np.nan)
    for x, y in product(np.arange(x_size), np.arange(y_size)):

        # Define shortcuts for the time series that we compare
        a = hypothesized_stack[:, x, y]
        b = residual_stack[:, x, y]

        # Skip pixels with NaN values
        if np.isnan(a).any() or np.isnan(b).any():
            continue

        # Compute the similarity between the expected signal and the residual
        similarities[x, y] = float(
            cosine_similarity(X=a.reshape(1, -1), Y=b.reshape(1, -1))
        )
    print('Done!', flush=True)

    print('Saving similarities to FITS...', end=' ', flush=True)
    file_path = results_dir / 'similarities.fits'
    save_fits(array=similarities, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
