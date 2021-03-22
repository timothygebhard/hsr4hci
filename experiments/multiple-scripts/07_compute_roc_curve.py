"""
Compute and plot the ROC curve for thresholding the match fraction.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, Union

import os
import time

from astropy.units import Quantity
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_erosion,
    generate_binary_structure
)
from sklearn.metrics import roc_curve

import h5py
import matplotlib.pyplot as plt
import numpy as np

from hsr4hci.utils.config import load_config, get_data_dir
from hsr4hci.utils.fits import read_fits
from hsr4hci.utils.masking import get_roi_mask
from hsr4hci.utils.units import set_units_for_instrument


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCOMPUTE ROC CURVES\n', flush=True)

    # -------------------------------------------------------------------------
    # Load experiment configuration and data
    # -------------------------------------------------------------------------

    # Define paths for experiment folder and results folder
    experiment_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    results_dir = experiment_dir / 'results'
    matches_dir = results_dir / 'matches'

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    file_path = experiment_dir / 'config.json'
    config = load_config(file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Load the binary mask that belongs to this data set
    # -------------------------------------------------------------------------

    # Construct file path
    file_path = Path(
        get_data_dir(),
        config['dataset']['target_name'],
        config['dataset']['filter_name'],
        config['dataset']['date'],
        'processed',
        f'stacked__{config["dataset"]["stacking_factor"]}.hdf',
    )

    # Read data from the HDF file
    with h5py.File(file_path, 'r') as hdf_file:

        # Get binary mask
        binary_mask = np.array(hdf_file['planet_paths_mask'])

        # Get metadata
        metadata: Dict[str, Union[str, float]] = dict()
        for key in hdf_file.attrs.keys():
            metadata[key] = hdf_file.attrs[key]

        # Get stack shape
        n_frames, x_size, y_size = np.array(hdf_file['stack']).shape

    # -------------------------------------------------------------------------
    # Define various useful shortcuts
    # -------------------------------------------------------------------------

    # Quantities related to the size of the data set
    frame_size = (x_size, y_size)
    center = (x_size / 2, y_size / 2)

    # Metadata of the data set
    pixscale = float(metadata['PIXSCALE'])
    lambda_over_d = float(metadata['LAMBDA_OVER_D'])

    # Experiment configuration
    n_test_positions = config['consistency_checks']['n_test_positions']
    metric_function = config['signal_masking']['metric_function']

    # -------------------------------------------------------------------------
    # Activate the unit conversions for this instrument
    # -------------------------------------------------------------------------

    set_units_for_instrument(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
        verbose=False,
    )

    # -------------------------------------------------------------------------
    # Set up the mask for the region of interest (ROI)
    # -------------------------------------------------------------------------

    # Define a mask for the ROI
    roi_mask = get_roi_mask(
        mask_size=frame_size,
        inner_radius=Quantity(*config['roi_mask']['inner_radius']),
        outer_radius=Quantity(*config['roi_mask']['outer_radius']),
    )

    # -------------------------------------------------------------------------
    # Compute selection mask for pixels which are included in the ROC curve
    # -------------------------------------------------------------------------

    dilated = np.copy(binary_mask)
    eroded = np.copy(binary_mask)

    structure = generate_binary_structure(2, 1)
    for i in range(2):
        dilated = binary_dilation(dilated, structure=structure)
    eroded = binary_erosion(eroded, structure=structure)

    selection_mask = np.logical_xor(dilated, eroded)
    selection_mask = np.logical_and(roi_mask, np.logical_not(selection_mask))

    # -------------------------------------------------------------------------
    # Load match fraction and compute ROC curve
    # -------------------------------------------------------------------------

    # Load match fraction from FITS
    file_path = matches_dir / f'match_fraction__{metric_function}.fits'
    match_fraction = read_fits(file_path)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(
        y_true=binary_mask[selection_mask],
        y_score=np.nan_to_num(match_fraction[selection_mask]),
    )

    # -------------------------------------------------------------------------
    # Create ROC curve plot and save it
    # -------------------------------------------------------------------------

    # Make sure that the plots directory exists
    plots_dir = results_dir / 'plots_and_snrs'
    plots_dir.mkdir(exist_ok=True)

    # Plot true positive rate (TPR) over false positive rate (FPR)
    plt.plot(fpr, tpr)

    # Set up plot options
    plt.gcf().set_size_inches(6, 6, forward=True)
    plt.xlim(-0.01, 0.5)
    plt.ylim(0.5, 1.01)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({metric_function})')
    plt.plot((0, 1), (0, 1), color='black', lw=1, ls='--')
    plt.grid(which='both', color='lightgray', ls='--')
    plt.tight_layout()

    # Save the plot
    file_path = plots_dir / f'roc_curve__{metric_function}.png'
    plt.savefig(file_path, dpi=300)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
