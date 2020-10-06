"""
Run consistency checks on best models and compute match fraction.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import List

import os
import time

from scipy.ndimage import median_filter
from tqdm.auto import tqdm

import h5py
import numpy as np

from hsr4hci.utils.config import load_config
from hsr4hci.utils.data import load_parang
from hsr4hci.utils.derotating import derotate_combine
from hsr4hci.utils.fits import read_fits, save_fits


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCONSTRUCT RESIDUALS BASED ON MATCH FRACTION\n', flush=True)

    # -------------------------------------------------------------------------
    # Load experiment configuration
    # -------------------------------------------------------------------------

    # Define paths for experiment folder and results folder
    experiment_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    results_dir = experiment_dir / 'results'

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    file_path = experiment_dir / 'config.json'
    config = load_config(file_path=file_path.as_posix())
    print('Done!', flush=True)

    # Define shortcuts to experiment configuration
    n_test_positions = config['consistency_checks']['n_test_positions']
    filter_size = config['consistency_checks']['filter_size']

    # -------------------------------------------------------------------------
    # Load residuals, match_fraction and parallactic angles; define ROI
    # -------------------------------------------------------------------------

    # Define path to results directory and load results HDF file
    print('Loading residuals from results.hdf...', end=' ', flush=True)
    file_path = results_dir / 'results.hdf'
    with h5py.File(file_path, 'r') as hdf_file:
        baseline_residuals = np.array(hdf_file['baseline']['residuals'])
        signal_masking_residuals = np.array(hdf_file['best']['residuals'])
    print('Done!', flush=True)

    # Define path to results directory and load results HDF file
    print('Loading match_fraction.fits...', end=' ', flush=True)
    file_path = results_dir / 'match_fraction.fits'
    match_fraction = np.asarray(read_fits(file_path=file_path.as_posix()))
    print('Done!', flush=True)

    # Load parallactic angles
    print('Loading parallactic angles...', end=' ', flush=True)
    parang = load_parang(**config['dataset'])
    print('Done!', flush=True)

    # Define mask that tells us for which pixels a signal_masking residual is
    # even available (for pixels not in this mask, we will use the baseline)
    signal_masking_mask = np.isnan(signal_masking_residuals[0])

    # Apply the signal masking mask to the match fraction to automatically get
    # all the masks below to respect the "no signal masking residual available"
    # region correctly
    match_fraction[signal_masking_mask] = np.nan

    # Additionally, also get the ROI mask for masking the signal estimates
    roi_mask = np.logical_not(np.isnan(baseline_residuals[0]))

    # -------------------------------------------------------------------------
    # Loop over different thresholds, compute masks and get signal estimates
    # -------------------------------------------------------------------------

    # Keep track of the masks and signal estimates that we generate
    thresholded_masks: List[np.ndarray] = []
    filtered_masks: List[np.ndarray] = []
    signal_estimates: List[np.ndarray] = []

    # Define thresholds: The possible values of the match fraction are
    # determined by `n_test_positions` (0/n, 1/n, ..., n/n); therefore, this
    # defines all threshold values that can produce different results.
    # This array (and the thresholding criterion below) are chosen such that
    # the first signal_estimate will contain the result for *always* using the
    # signal masking-based residual (where available), while the last one will
    # contain the pure baseline residual.
    thresholds = np.linspace(0, 1, n_test_positions + 1)
    thresholds = np.insert(thresholds, 0, -1)

    print('')
    print('Running for the following thresholds:')
    print(thresholds)
    print('')

    print('Creating masks and signal estimates for thresholds:', flush=True)
    for threshold in tqdm(thresholds, ncols=80):

        # Apply thresholding and median filter to create binary masks for
        # selecting the residual type
        thresholded_mask = match_fraction > threshold
        filtered_mask = np.logical_and(
            median_filter(thresholded_mask, filter_size), thresholded_mask
        )

        # Store these masks
        thresholded_masks.append(thresholded_mask)
        filtered_masks.append(filtered_mask)

        # Initialize residuals: by default, use baseline model for everything
        residuals = np.copy(baseline_residuals)

        # Only for the pixels selected by the filtered mask do we choose the
        # residuals from the best model based on signal masking
        residuals[:, filtered_mask] = \
            signal_masking_residuals[:, filtered_mask]

        # Compute signal estimate and store it
        signal_estimate = derotate_combine(residuals, parang)
        signal_estimate[~roi_mask] = np.nan
        signal_estimates.append(signal_estimate)

    print('')

    # -------------------------------------------------------------------------
    # Save residuals and signal estimate as FITS files
    # -------------------------------------------------------------------------

    print('Saving thresholded masks to FITS...', end=' ', flush=True)
    array = np.array(thresholded_masks).astype(int)
    file_path = results_dir / 'thresholded_masks.fits'
    save_fits(array=array, file_path=file_path)
    print('Done!', flush=True)

    print('Saving filtered masks to FITS...', end=' ', flush=True)
    array = np.array(filtered_masks).astype(int)
    file_path = results_dir / 'filtered_masks.fits'
    save_fits(array=array, file_path=file_path.as_posix())
    print('Done!', flush=True)

    print('Saving signal estimates to FITS...', end=' ', flush=True)
    array = np.array(signal_estimates)
    file_path = results_dir / 'signal_estimates.fits'
    save_fits(array=array, file_path=file_path.as_posix())
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
