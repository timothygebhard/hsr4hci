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

import numpy as np

from hsr4hci.utils.config import load_config
from hsr4hci.utils.data import load_data
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

    # Add optional arguments
    parser.add_argument('--x',
                        type=int,
                        metavar='X',
                        help='x-coordinate')
    parser.add_argument('--y',
                        type=int,
                        metavar='Y',
                        help='y-coordinate')

    # Parse the command line arguments and return the result
    return parser.parse_args()


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nTRAIN HALF-SIBLING REGRESSION MODEL FOR A POSITION\n', flush=True)

    # -------------------------------------------------------------------------
    # Load config and data
    # -------------------------------------------------------------------------

    # Get command line options
    args = get_arguments()
    position = (int(args.x), int(args.y))

    # Load experiment config from JSON
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    config = load_config(os.path.join(experiment_dir, 'config.json'))

    # Load frames and parallactic angles from HDF file
    stack, parang, psf_template = load_data(dataset_config=config['dataset'])

    # Print some basic training information
    print(f'Stack size:\t {stack.shape}', flush=True)
    print(f'Model type:\t {config["experiment"]["model"]["module"]}.'
          f'{config["experiment"]["model"]["class"]}', flush=True)

    # -------------------------------------------------------------------------
    # Set up and train model
    # -------------------------------------------------------------------------

    # Instantiate model
    hsr = HalfSiblingRegression(config=config)

    # Train the model for all pixels in the ROI
    print(f'\nTraining models for position {position}:', flush=True)
    hsr.train_position(position=position,
                       stack=stack,
                       parang=parang,
                       psf_template=psf_template)

    # -------------------------------------------------------------------------
    # Ensure the results dir exists
    # -------------------------------------------------------------------------

    results_dir = os.path.join(experiment_dir, 'results')
    Path(results_dir).mkdir(exist_ok=True)

    positions_dir = os.path.join(results_dir, 'positions')
    Path(positions_dir).mkdir(exist_ok=True)

    detection_maps_dir = os.path.join(positions_dir, 'detection_maps')
    Path(detection_maps_dir).mkdir(exist_ok=True)

    residuals_dir = os.path.join(positions_dir, 'residuals')
    Path(residuals_dir).mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Get the residual stack and save it
    # -------------------------------------------------------------------------

    # Get residual stack
    print(f'\nComputing residual stack:', flush=True)
    residual_stack = hsr.get_residual_stack(stack=stack)
    residuals = residual_stack[:, position[0], position[1]].astype(np.float32)

    # Save the residuals as a numpy array
    print(f'\nSaving residuals to numpy format...', end=' ', flush=True)
    file_path = os.path.join(residuals_dir,
                             f'residuals__{args.x}_{args.y}.npy')
    np.save(file=file_path, arr=residuals)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
