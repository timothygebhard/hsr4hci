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

    # Add arguments
    parser.add_argument('--n-regions',
                        type=int,
                        metavar='N',
                        help='Number of regions in total')
    parser.add_argument('--region-idx',
                        type=int,
                        metavar='N',
                        help='Index of the region for which to run')

    # Parse and store the command line arguments
    args = parser.parse_args()

    # Run sanity checks on arguments
    if not 0 <= args.region_idx < args.n_regions:
        raise ValueError('region_idx must be between 0 and n_regions-1!')

    return args


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

    # Train the model for all pixels in the region specified in the args
    print(f'\nTraining models for region {region_idx + 1} of {n_regions}:')
    hsr.train_region(region_idx=region_idx,
                     n_regions=n_regions,
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
    residual_stack = hsr.get_residual_stack(stack=stack).astype(np.float32)

    # Loop over all positions for which we computed residuals and store them
    # separately as numpy arrays (to minimize the amount of wasted memory)
    print(f'\nSaving residuals to numpy format:', flush=True)
    for position, _ in tqdm(hsr.m__collections.items(), ncols=80):

        # Select the residuals for the current position
        residuals = residual_stack[:, position[0], position[1]]

        # Save the residuals as a numpy array
        
        file_path = os.path.join(residuals_dir,
                                 f'residuals__{position[0]}_{position[1]}.npy')
        np.save(file=file_path, arr=residuals)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
