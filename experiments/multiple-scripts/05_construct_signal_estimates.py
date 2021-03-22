"""
Loop over thresholds to compute selection masks; assemble residuals
based on these selection masks and compute signal estimates from them.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import os
import time

from skimage.filters import threshold_minimum

import numpy as np

from hsr4hci.utils.config import load_config
from hsr4hci.utils.data import load_parang
from hsr4hci.utils.fits import read_fits, save_fits
from hsr4hci.utils.general import find_closest
from hsr4hci.utils.hdf import load_dict_from_hdf
from hsr4hci.utils.signal_estimates import get_signal_estimates_and_masks
from hsr4hci.utils.signal_masking import assemble_signal_masking_residuals


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

    # -------------------------------------------------------------------------
    # Load residuals, match_fraction and parallactic angles; define ROI
    # -------------------------------------------------------------------------

    # Load hypotheses from FITS file
    print('Loading hypotheses from FITS file...', end=' ', flush=True)
    file_path = hypotheses_dir / 'hypotheses.fits'
    hypotheses = np.asarray(read_fits(file_path=file_path))
    print('Done!', flush=True)

    # Load results from HDF file
    print('Loading main results file...', end=' ', flush=True)
    file_path = results_dir / 'results.hdf'
    results = load_dict_from_hdf(file_path=file_path)
    print('Done!', flush=True)

    # Load match fraction from FITS file
    print('Loading match_fraction.fits...', end=' ', flush=True)
    file_path = matches_dir / 'match_fraction__median.fits'
    match_fraction = np.asarray(read_fits(file_path=file_path))
    print('Done!', flush=True)

    # Load parallactic angles from HDF file
    print('Loading parallactic angles...', end=' ', flush=True)
    parang = load_parang(**config['dataset'])
    print('Done!', flush=True)

    # Define various useful shortcuts
    n_frames = len(parang)
    x_size, y_size = hypotheses.shape

    # -------------------------------------------------------------------------
    # Assemble residuals based on hypotheses; get ROI mask
    # -------------------------------------------------------------------------

    # Select residuals for default case and for signal masking
    print('Assembling residuals...', end=' ', flush=True)
    default_residuals = results['default']['residuals']
    signal_masking_residuals = assemble_signal_masking_residuals(
        hypotheses=hypotheses, results=results
    )
    print('Done!\n', flush=True)

    # Get the ROI mask for masking the signal estimates
    roi_mask = np.logical_not(np.isnan(default_residuals[0]))

    # -------------------------------------------------------------------------
    # Loop over different thresholds, compute masks and get signal estimates
    # -------------------------------------------------------------------------

    print('Creating masks and signal estimates for thresholds:', flush=True)
    (signal_estimates,
     thresholds,
     thresholded_masks,
     filtered_masks) = get_signal_estimates_and_masks(
        parang=parang,
        match_fraction=match_fraction,
        default_residuals=default_residuals,
        signal_masking_residuals=signal_masking_residuals,
        roi_mask=roi_mask,
        filter_size=config['consistency_checks']['filter_size'],
        n_thresholds=100,
    )
    print('')

    # Determine a single "good" threshold
    good_threshold = threshold_minimum(np.nan_to_num(match_fraction))
    idx, _ = find_closest(sequence=thresholds, value=good_threshold)
    signal_estimate = signal_estimates[idx]
    print(f'"Good" threshold: {good_threshold:.2f}\n')

    # -------------------------------------------------------------------------
    # Save residuals and signal estimate as FITS files
    # -------------------------------------------------------------------------

    # Create directory for saving the selection masks
    masks_dir = results_dir / 'masks_and_thresholds'
    masks_dir.mkdir(exist_ok=True)

    print('Saving thresholded masks to FITS...', end=' ', flush=True)
    array = np.array(thresholded_masks).astype(int)
    file_path = masks_dir / 'thresholded_masks.fits'
    save_fits(array=array, file_path=file_path)
    print('Done!', flush=True)

    print('Saving filtered masks to FITS...', end=' ', flush=True)
    array = np.array(filtered_masks).astype(int)
    file_path = masks_dir / 'filtered_masks.fits'
    save_fits(array=array, file_path=file_path)
    print('Done!', flush=True)

    print('Saving thresholds to CSV file...', end=' ', flush=True)
    file_path = masks_dir / 'thresholds.csv'
    thresholds.tofile(file_path, sep=',', format='%10.5f')
    print('Done!', flush=True)

    print('Saving signal estimates to FITS...', end=' ', flush=True)
    array = np.array(signal_estimates)
    file_path = results_dir / 'signal_estimates.fits'
    save_fits(array=array, file_path=file_path)
    print('Done!', flush=True)

    print('Saving "best" signal estimate to FITS...', end=' ', flush=True)
    file_path = results_dir / 'signal_estimate.fits'
    save_fits(array=signal_estimate, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
