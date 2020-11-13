"""
Loop over thresholds to compute selection masks; assemble residuals
based on these selection masks and compute signal estimates from them.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from itertools import product
from pathlib import Path
from typing import List

import os
import time

from skimage.morphology import disk, opening
from tqdm.auto import tqdm

import numpy as np

from hsr4hci.utils.config import load_config
from hsr4hci.utils.data import load_parang
from hsr4hci.utils.derotating import derotate_combine
from hsr4hci.utils.fits import read_fits, save_fits
from hsr4hci.utils.hdf import load_dict_from_hdf


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

    # Define paths for experiment folder and result folders
    experiment_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    results_dir = experiment_dir / 'results'
    hypotheses_dir = results_dir / 'hypotheses'
    matches_dir = results_dir / 'matches'

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    file_path = experiment_dir / 'config.json'
    config = load_config(file_path=file_path)
    print('Done!', flush=True)

    # Define shortcuts to experiment configuration
    filter_size = config['consistency_checks']['filter_size']
    metric_function = config['signal_masking']['metric_function']
    n_test_positions = config['consistency_checks']['n_test_positions']

    # -------------------------------------------------------------------------
    # Load residuals, match_fraction and parallactic angles; define ROI
    # -------------------------------------------------------------------------

    # Load hypotheses from FITS file
    print('Loading hypotheses from FITS file...', end=' ', flush=True)
    file_path = hypotheses_dir / f'hypotheses__{metric_function}.fits'
    hypotheses = np.asarray(read_fits(file_path=file_path))
    print('Done!', flush=True)

    # Load results from HDF file
    print('Loading main results file...', end=' ', flush=True)
    file_path = results_dir / 'results.hdf'
    results = load_dict_from_hdf(file_path=file_path)
    print('Done!', flush=True)

    # Load match fraction from FITS file
    print('Loading match_fraction.fits...', end=' ', flush=True)
    file_path = matches_dir / f'match_fraction__{metric_function}.fits'
    match_fraction = np.asarray(read_fits(file_path=file_path))
    print('Done!', flush=True)

    # Load parallactic angles from HDF file
    print('Loading parallactic angles...', end=' ', flush=True)
    parang = load_parang(**config['dataset'])
    print('Done!\n', flush=True)

    # Define various useful shortcuts
    n_frames = len(parang)
    x_size, y_size = hypotheses.shape

    # -------------------------------------------------------------------------
    # Assemble residuals based on hypotheses; get ROI mask
    # -------------------------------------------------------------------------

    # Select default model residuals (i.e., from models without signal masking)
    default_residuals = results['default']['residuals']

    # Assemble signal masking residuals based on the hypotheses: for each
    # spatial position, we take the "best" residual according to the signal
    # time that is stored in the hypotheses array
    print('Assembling signal masking residuals...', end=' ', flush=True)
    signal_masking_residuals = np.full((n_frames, x_size, y_size), np.nan)
    for x, y in product(range(x_size), range(y_size)):
        if not np.isnan(signal_time := hypotheses[x, y]):
            signal_masking_residuals[:, x, y] = np.array(
                results[str(int(signal_time))]['residuals'][:, x, y]
            )
    print('Done!\n', flush=True)

    # Get the ROI mask for masking the signal estimates
    roi_mask = np.logical_not(np.isnan(default_residuals[0]))

    # -------------------------------------------------------------------------
    # Loop over different thresholds, compute masks and get signal estimates
    # -------------------------------------------------------------------------

    # Define threshold values
    thresholds = np.linspace(0, 1, 51)
    thresholds = np.insert(thresholds, -1, [-1, 0, 1])
    thresholds = np.array(sorted(np.unique(thresholds)))
    print('Running for the following thresholds:\n', thresholds, '\n')

    # Keep track of the masks and signal estimates that we generate
    thresholded_masks: List[np.ndarray] = []
    filtered_masks: List[np.ndarray] = []
    signal_estimates: List[np.ndarray] = []

    print('Creating masks and signal estimates for thresholds:', flush=True)
    for threshold in tqdm(thresholds, ncols=80):

        # Threshold the matching fraction
        thresholded_mask = match_fraction > threshold
        thresholded_masks.append(thresholded_mask)

        # Define a structure element and apply a morphological filter (more
        # precisely, an opening filter) to remove small regions in the mask.
        # This reflects our knowledge that a true planet path should not have
        # the characteristic "sausage"-shape, and not consist of single pixels
        structure_element = disk(filter_size)
        filtered_mask = np.logical_and(
            opening(thresholded_mask, structure_element), thresholded_mask
        )
        filtered_masks.append(filtered_mask)

        # Select the residuals: by default, use default model for everything.
        # Only for the pixels selected by the filtered mask do we choose the
        # residuals from the best model based on signal masking.
        residuals = np.copy(default_residuals)
        residuals[:, filtered_mask] = np.array(
            signal_masking_residuals[:, filtered_mask]
        )

        # Compute signal estimate and store it
        signal_estimate = derotate_combine(residuals, parang)
        signal_estimate[~roi_mask] = np.nan
        signal_estimates.append(signal_estimate)

    print('')

    # -------------------------------------------------------------------------
    # Save residuals and signal estimate as FITS files
    # -------------------------------------------------------------------------

    # Create directory for saving the selection masks
    masks_dir = results_dir / 'masks_and_thresholds'
    masks_dir.mkdir(exist_ok=True)

    print('Saving thresholded masks to FITS...', end=' ', flush=True)
    array = np.array(thresholded_masks).astype(int)
    file_path = masks_dir / f'thresholded_masks__{metric_function}.fits'
    save_fits(array=array, file_path=file_path)
    print('Done!', flush=True)

    print('Saving filtered masks to FITS...', end=' ', flush=True)
    array = np.array(filtered_masks).astype(int)
    file_path = masks_dir / f'filtered_masks__{metric_function}.fits'
    save_fits(array=array, file_path=file_path)
    print('Done!', flush=True)

    print('Saving thresholds to CSV file...', end=' ', flush=True)
    file_path = masks_dir / f'thresholds__{metric_function}.csv'
    thresholds.tofile(file_path, sep=',', format='%10.5f')
    print('Done!', flush=True)

    print('Saving signal estimates to FITS...', end=' ', flush=True)
    array = np.array(signal_estimates)
    file_path = results_dir / f'signal_estimates__{metric_function}.fits'
    save_fits(array=array, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
