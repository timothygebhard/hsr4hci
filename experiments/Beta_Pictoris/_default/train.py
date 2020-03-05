"""
Train a HalfSiblingRegression model (by default iteratively).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import os
import time

import numpy as np

from hsr4hci.utils.config import load_config
from hsr4hci.utils.data import load_data
from hsr4hci.utils.em import get_signal_estimate_stack, relu
from hsr4hci.utils.fits import save_fits
from hsr4hci.models.hsr import HalfSiblingRegression


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nTRAIN HALF-SIBLING REGRESSION MODEL\n', flush=True)

    # -------------------------------------------------------------------------
    # Load config and data
    # -------------------------------------------------------------------------

    # Load experiment config from JSON
    # NOTE: This also activates the instrument-specific unit conversions
    # based on the values for the pixel scale and lambda_over_d.
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    config = load_config(os.path.join(experiment_dir, 'config.json'))

    # Load frames and parallactic angles from HDF file
    stack, parang, _ = load_data(**config['dataset'])

    # Print some general information
    print('Stack:', stack.shape)
    print('ROI:  ', ' - '.join([str(_) for _ in config['roi_mask'].values()]))

    # -------------------------------------------------------------------------
    # Run the HSR multiple times, in a EM-style iterative procedure
    # -------------------------------------------------------------------------

    # Get the number of iterations from the config file
    n_iterations = config['em']['n_iterations']

    # Initialize the signal_estimate
    signal_estimate = np.zeros(stack.shape[1:])

    train_mask = None

    # Run for the given number of iterations
    for iteration in range(n_iterations):

        print(f'\n\nITERATION {iteration + 1} / {n_iterations}')

        # Define results directory for this iteration
        results_dir = os.path.join(config['experiment_dir'], 'results',
                                   f'round_{iteration:03}')
        Path(results_dir).mkdir(exist_ok=True, parents=True)

        # Construct the input stack for the half-sibling regression by
        # removing the current signal estimate from the raw stack
        signal_estimate = relu(signal_estimate)
        signal_estimate_stack = \
            get_signal_estimate_stack(signal_estimate=signal_estimate,
                                      parang=parang)
        input_stack = stack - signal_estimate_stack

        # Save the signal_estimate_stack (for debugging purposes)
        file_path = os.path.join(results_dir, f'signal_estimate_stack.fits')
        save_fits(array=signal_estimate_stack, file_path=file_path)

        # Save the input stack (for debugging purposes)
        file_path = os.path.join(results_dir, f'input_stack.fits')
        save_fits(array=input_stack, file_path=file_path)

        # Instantiate a new HSR model for this iteration
        hsr = HalfSiblingRegression(config=config,
                                    results_dir=results_dir,
                                    train_mask=train_mask,
                                    **config['debugging'])

        # Train the model on the input stack for this iteration
        print('Training model:', flush=True)
        hsr.train(stack=input_stack,
                  parang=parang)

        # Save the immediate results
        print()
        hsr.save_residuals()
        hsr.save_predictions()
        hsr.save_r_squared()
        hsr.save_coefficients()

        # Update the signal estimate by adding the "delta residual", that is,
        # the signal estimate from this last round
        delta_residual = hsr.get_signal_estimate(parang=parang,
                                                 subtract=None,
                                                 combine='mean')
        signal_estimate += np.nan_to_num(delta_residual)

        # Save delta residual
        print('Saving delta residual...', end=' ', flush=True)
        file_path = os.path.join(results_dir, 'delta_residual.fits')
        save_fits(array=delta_residual, file_path=file_path)
        print('Done!', flush=True)

        # Save new signal estimate (and apply ROI mask before saving)
        print('Saving signal estimate...', end=' ', flush=True)
        signal_estimate_ = np.copy(signal_estimate)
        signal_estimate_[~hsr.roi_mask] = np.nan
        file_path = os.path.join(results_dir, 'signal_estimate.fits')
        save_fits(array=signal_estimate_, file_path=file_path)
        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
