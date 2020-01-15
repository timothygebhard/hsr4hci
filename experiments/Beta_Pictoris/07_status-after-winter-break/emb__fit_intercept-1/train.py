"""
Train an HSR model (by default with forward modeling).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import os
import time

from tqdm import tqdm

import numpy as np

from hsr4hci.utils.adi_tools import derotate_frames
from hsr4hci.utils.config import load_config
from hsr4hci.utils.data import load_data
from hsr4hci.utils.fits import read_fits, save_fits
from hsr4hci.models.hsr import HalfSiblingRegression


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
    parser.add_argument('--n-regions',
                        type=int,
                        metavar='N',
                        help='Number of regions in total')
    parser.add_argument('--region-idx',
                        type=int,
                        metavar='N',
                        help='Index of the region for which to run')
    parser.add_argument('--emb-round',
                        type=int,
                        metavar='N',
                        help='Iteration of the EMB procedure')

    # Parse and store the command line arguments
    args = parser.parse_args()

    # Run sanity checks on arguments
    if not 0 <= args.region_idx < args.n_regions:
        raise ValueError('region_idx must be between 0 and n_regions-1!')

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
    print('\nTRAIN HALF-SIBLING REGRESSION MODEL FOR A REGION\n', flush=True)

    # -------------------------------------------------------------------------
    # Load config and data
    # -------------------------------------------------------------------------

    # Get command line options
    args = get_arguments()
    n_regions = int(args.n_regions)
    region_idx = int(args.region_idx)
    emb_round = int(args.emb_round)

    # Load experiment config from JSON
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    config = load_config(os.path.join(experiment_dir, 'config.json'))

    # Load frames and parallactic angles from HDF file
    raw_stack, parang, psf_template = \
        load_data(dataset_config=config['dataset'])

    # Print some basic training information
    print(f'Stack size:\t {raw_stack.shape}', flush=True)
    print(f'Model type:\t {config["experiment"]["model"]["module"]}.'
          f'{config["experiment"]["model"]["class"]}', flush=True)

    # -------------------------------------------------------------------------
    # Ensure the results dir exists
    # -------------------------------------------------------------------------

    results_dir = os.path.join(experiment_dir, 'results')
    Path(results_dir).mkdir(exist_ok=True)

    round_dir = os.path.join(results_dir, f'round_{emb_round:03}')
    Path(round_dir).mkdir(exist_ok=True)

    positions_dir = os.path.join(round_dir, 'positions')
    Path(positions_dir).mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Get the last planet signal and noise estimates
    # -------------------------------------------------------------------------

    # In the first round, we do not have a (planet) signal estimate yet; thus,
    # we initialize it as zeros
    if emb_round == 0:
        last_signal_estimate = np.zeros(raw_stack.shape[1:])
        last_noise_estimate_stack = np.zeros_like(raw_stack)

    # Otherwise, we load the signal estimate from the last round
    else:

        # Define directory to last round
        last_round_dir = os.path.join(experiment_dir, 'results',
                                      f'round_{(emb_round - 1):03}')

        # Load last signal estimate
        file_path = os.path.join(last_round_dir, 'signal_estimate.fits')
        last_signal_estimate = read_fits(file_path=file_path)

        # Load last noise estimate
        file_path = os.path.join(last_round_dir, 'noise_estimate_stack.fits')
        last_noise_estimate_stack = read_fits(file_path=file_path)

    # Pass the signal estimate through a ReLU function
    last_signal_estimate = relu(last_signal_estimate)

    # Construct a signal stack from it and rotate it into the coordinate system
    # of the original stack from which we are going to subtract it
    signal_estimate_stack = \
        [last_signal_estimate for _ in range(len(raw_stack))]
    last_signal_estimate_stack = np.array(signal_estimate_stack)
    last_signal_estimate_stack = \
        derotate_frames(stack=last_signal_estimate_stack,
                        parang=(parang[0]-parang))

    # Get the input stack on which we will train the model
    # A_i = Y - ReLU( P_{i-1} )
    stack_without_planet = raw_stack - last_signal_estimate_stack

    # B_i = A_i - N_{i-1}
    input_stack = stack_without_planet - last_noise_estimate_stack

    # Save the input_stack (B_i) so that we have it available in the
    # merge_residuals.py
    if region_idx == 0:
        file_path = os.path.join(round_dir, 'input_stack.fits')
        save_fits(array=input_stack, file_path=file_path)

    # -------------------------------------------------------------------------
    # Set up and train model
    # -------------------------------------------------------------------------

    # Instantiate model
    hsr = HalfSiblingRegression(config=config)

    # Train the model for all pixels in the region specified in the args
    print(f'\nTraining models for region {region_idx + 1} of {n_regions}:')
    hsr.train_region(region_idx=region_idx,
                     n_regions=n_regions,
                     stack=input_stack,
                     parang=parang,
                     psf_template=psf_template)

    # -------------------------------------------------------------------------
    # Get the residual stack and save it
    # -------------------------------------------------------------------------

    # Get residual stack
    # Q_i = HSR(B_i)
    print(f'\nComputing residual stack:', flush=True)
    residual_stack = hsr.get_residual_stack(stack=input_stack)
    residual_stack = residual_stack.astype(np.float32)

    # Loop over all positions for which we computed residuals and store them
    # separately as numpy arrays (to minimize the amount of wasted memory)
    print(f'\nSaving residuals to numpy format:', flush=True)
    for position, _ in tqdm(hsr.m__collections.items(), ncols=80):

        # Select the residuals for the current position
        residuals = residual_stack[:, position[0], position[1]]

        # Save the residuals as a numpy array
        file_path = os.path.join(positions_dir,
                                 f'residuals__{position[0]}_{position[1]}.npy')
        np.save(file=file_path, arr=residuals)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
