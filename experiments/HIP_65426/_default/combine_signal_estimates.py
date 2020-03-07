"""
Combine signal estimates from different rounds into a single FITS file.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import time

import numpy as np

from hsr4hci.utils.fits import read_fits, save_fits


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCOMBINE SIGNAL ESTIMATES\n', flush=True)

    # -------------------------------------------------------------------------
    # Collect signal estimates from every round
    # -------------------------------------------------------------------------

    # Define paths to experiment and results dir
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    results_dir = os.path.join(experiment_dir, 'results')

    # Store the signal estimate for every round
    signal_estimates = list()

    # Loop over all "possible" rounds (this is a somewhat hacky but easy way
    # to collect all rounds without having to know the exact number in advance)
    print('Collecting signal estimates...', end=' ', flush=True)
    for em_round in range(1000):

        # Construct expected path to FITS file
        round_dir = os.path.join(results_dir, f'round_{em_round:03}')
        file_path = os.path.join(round_dir, 'signal_estimate.fits')

        # If the file exists, read it in and store it. Otherwise we can break
        # the loop here, because we've reached the maximum round.
        if os.path.exists(file_path):
            signal_estimate = read_fits(file_path=file_path)
            signal_estimates.append(signal_estimate)
        else:
            break
    print(f'Done! (found {len(signal_estimates)} FITS files)', flush=True)

    # Convert the collected results to a 3D numpy array, where the first axis
    # corresponds to the round of the EM algorithm
    signal_estimates = np.array(signal_estimates)

    # -------------------------------------------------------------------------
    # Save the combined results
    # -------------------------------------------------------------------------

    print('Saving combined signal estimates...', end=' ', flush=True)
    file_path = os.path.join(results_dir, 'combined_signal_estimates.fits')
    save_fits(array=signal_estimates, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
