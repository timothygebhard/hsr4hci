"""
Merge residuals that were generated on the cluster in parallel.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import os
import time

from tqdm import tqdm

import numpy as np

from hsr4hci.utils.adi_tools import derotate_frames
from hsr4hci.utils.config import load_config
from hsr4hci.utils.data import load_data
from hsr4hci.utils.fits import read_fits, save_fits


# -----------------------------------------------------------------------------
# AUXILIARY FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_arguments() -> argparse.Namespace:
    """
    Parse and return the command line arguments.

    Returns:
        An `argparse.Namespace` object containing the command line
        options that were passed to this script.
    """

    # Set up a parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--emb-round',
                        type=int,
                        metavar='N',
                        help='Iteration of the EMB procedure')

    # Parse and store the command line arguments
    args = parser.parse_args()

    return args


def relu(x):
    return np.maximum(x, 0)


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nMERGE RESIDUALS\n', flush=True)

    # -------------------------------------------------------------------------
    # Load experiment config and get expected size of the residual stack
    # -------------------------------------------------------------------------

    # Get command line arguments
    args = get_arguments()
    emb_round = int(args.emb_round)

    # Load experiment config from JSON
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    config = load_config(os.path.join(experiment_dir, 'config.json'))

    # Load frames and parallactic angles from HDF file
    raw_stack, parang, psf_template = \
        load_data(dataset_config=config['dataset'])

    # Get the spatial size of the residual stack that we are assembling
    frame_size = tuple(config['dataset']['frame_size'])

    # -------------------------------------------------------------------------
    # Collect all numpy files with residuals for the current round
    # -------------------------------------------------------------------------

    # Define shortcuts to different result directories
    results_dir = os.path.join(experiment_dir, 'results')
    round_dir = os.path.join(results_dir, f'round_{emb_round:03}')
    positions_dir = os.path.join(round_dir, 'positions')

    # Get all numpy files in the "positions"-directory
    print('Collecting files in residuals directory...', end=' ', flush=True)
    numpy_files = [_ for _ in os.listdir(positions_dir) if _.endswith('npy')]
    print(f'Done! ({len(numpy_files)} files found)\n', flush=True)

    # Collection detection maps that we read in
    residual_stack = None

    # Loop over all files, read them in and add residuals to the residual_stack
    print('Reading in numpy files from results directory:')
    for numpy_file in tqdm(numpy_files, ncols=80):

        # Get the position that the current file corresponds to
        position = \
            tuple(map(int, numpy_file.split('__')[1].split('.')[0].split('_')))

        # Read in the numpy file
        file_path = os.path.join(positions_dir, numpy_file)
        array = np.load(file=file_path)

        # Once we have read in the first array, we finally know the shape of
        # the full residual stack and can properly initialize it
        if residual_stack is None:
            stack_shape = (len(array), ) + frame_size
            residual_stack = np.full(stack_shape, np.nan)

        # Add the array containing the residuals for the current position into
        # the residual_stack (at that position)
        residual_stack[:, position[0], position[1]] = array

    print('')

    # -------------------------------------------------------------------------
    # If we are not using a forward model: de-rotate the stack
    # -------------------------------------------------------------------------

    # If we do not use forward modeling, the residual stack we combine needs
    # to be de-rotated first such that we can compute the signal_estimate etc.
    if not config['experiment']['use_forward_model']:
        residual_stack = derotate_frames(stack=residual_stack,
                                         parang=(parang-parang[0]))

    # -------------------------------------------------------------------------
    # Save the full residual stack
    # -------------------------------------------------------------------------

    print('Saving residual stack...', end=' ', flush=True)
    file_path = os.path.join(round_dir, f'residual_stack.fits')
    save_fits(residual_stack, file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Compute delta residual and save it
    # -------------------------------------------------------------------------

    print('Computing delta residual...', end=' ', flush=True)

    with np.warnings.catch_warnings():

        # Suppress "All-NaN slice / axis encountered" warning, which is
        # expected for pixels outside of the region of interest
        np.warnings.filterwarnings('ignore', r'All-NaN \w* encountered')

        # Compute the delta residual
        delta_residual = np.nanmean(residual_stack, axis=0)

        # Center the delta residual (ensure mean=0)
        if emb_round > 0:
            delta_residual -= np.nanmean(delta_residual)

    print('Done!', flush=True)

    print('Saving delta residual...', end=' ', flush=True)
    file_path = os.path.join(round_dir, f'delta_residual.fits')
    save_fits(delta_residual, file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Compute new signal estimate and save it
    # -------------------------------------------------------------------------

    print('Computing new signal estimate...', end=' ', flush=True)

    # Load the last signal estimate
    if emb_round == 0:
        last_signal_estimate = np.zeros(frame_size)
    else:
        last_round_dir = os.path.join(experiment_dir, 'results',
                                      f'round_{(emb_round - 1):03}')
        file_path = os.path.join(last_round_dir, 'signal_estimate.fits')
        last_signal_estimate = read_fits(file_path=file_path)

    # Pass the last signal estimate through a ReLU function
    last_signal_estimate = relu(last_signal_estimate)

    # Compute the full residual
    signal_estimate = last_signal_estimate + delta_residual

    print('Done!', flush=True)

    print('Saving new signal estimate...', end=' ', flush=True)
    file_path = os.path.join(round_dir, f'signal_estimate.fits')
    save_fits(signal_estimate, file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Compute current noise estimate and save it
    # -------------------------------------------------------------------------

    print('Computing noise estimate...', end=' ', flush=True)

    # Load input_stack (B_i) from this round
    file_path = os.path.join(round_dir, 'input_stack.fits')
    input_stack = read_fits(file_path=file_path)

    # Load noise_estimate_stack from last round
    if emb_round == 0:
        last_noise_estimate_stack = np.zeros_like(raw_stack)
    else:
        last_round_dir = os.path.join(experiment_dir, 'results',
                                      f'round_{(emb_round - 1):03}')
        file_path = os.path.join(last_round_dir, 'noise_estimate_stack.fits')
        last_noise_estimate_stack = read_fits(file_path=file_path)

    # Compute new noise estimate
    # N_i = N_{i-1} + B_i - Derotate(residual_stack_{i})
    residual_stack_derotated = derotate_frames(stack=residual_stack,
                                               parang=(parang[0]-parang))

    # Boost (i.e., update the noise estimate) every third EMB round
    if (emb_round % 3) == 2:
        noise_estimate_stack = \
            last_noise_estimate_stack + input_stack - residual_stack_derotated
    else:
        noise_estimate_stack = last_noise_estimate_stack

    # Save the noise_estimate_stack
    file_path = os.path.join(round_dir, 'noise_estimate_stack.fits')
    save_fits(array=noise_estimate_stack, file_path=file_path)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
