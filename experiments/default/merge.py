"""
Merge (derotated) frames into a single residual frame by averaging.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import copy
import h5py
import numpy as np
import os
import time

from hsr4hci.utils.adi_tools import derotate_frames
from hsr4hci.utils.config import load_config
from hsr4hci.utils.evaluation import compute_figures_of_merit
from hsr4hci.utils.fits import save_fits
from hsr4hci.utils.roi_selection import get_roi_mask


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nMERGE AND EVALUATE RESULTS\n', flush=True)

    # -------------------------------------------------------------------------
    # Load config and define shortcuts
    # -------------------------------------------------------------------------

    # Load experiment config from JSON
    config = load_config('./config.json')

    # Define shortcuts for performance evaluation
    position = (config['experiment']['evaluation']['planet_position']['x'],
                config['experiment']['evaluation']['planet_position']['y'])
    aperture_size = config['experiment']['evaluation']['aperture_size']
    ignore = config['experiment']['evaluation']['ignore_neighbors']
    optimize = config['experiment']['evaluation']['optimize']

    # -------------------------------------------------------------------------
    # Get ROI mask
    # -------------------------------------------------------------------------

    # Define shortcuts for ROI mask
    pixscale = config['dataset']['pixscale']
    roi_ier = config['experiment']['roi']['inner_exclusion_radius']
    roi_oer = config['experiment']['roi']['outer_exclusion_radius']
    mask_size = (int(config['dataset']['x_size']),
                 int(config['dataset']['y_size']))

    # Get mask for the region of interest
    roi_mask = get_roi_mask(mask_size=mask_size,
                            pixscale=pixscale,
                            inner_exclusion_radius=roi_ier,
                            outer_exclusion_radius=roi_oer)

    # -------------------------------------------------------------------------
    # Load data (predictions and residuals)
    # -------------------------------------------------------------------------

    # Define path to results directory
    results_dir = os.path.join(config['experiment_dir'], 'results')

    # Read in predictions, residuals and parallactic angles from HDF file
    print('Reading results from HDF...', end=' ', flush=True)
    hdf_file_path = os.path.join(results_dir, 'results.hdf')
    with h5py.File(hdf_file_path, 'r') as hdf_file:
        predictions = np.array(hdf_file['predictions'])
        residuals = np.array(hdf_file['residuals'])
        parang = np.array(hdf_file['parang'])
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Merge and evaluate
    # -------------------------------------------------------------------------

    for presub in ('none', 'mean', 'median'):
        for derotate in (True, False):
            for name, original_stack in (('predictions', predictions),
                                         ('residuals', residuals)):

                # Create a copy of the stack
                stack = copy.deepcopy(original_stack)

                # Perform pre-subtraction of mean / median
                with np.errstate(all='ignore'):
                    if presub == 'mean':
                        stack = stack - np.mean(stack, axis=0)
                    elif presub == 'median':
                        stack = stack - np.median(stack, axis=0)

                # Derotate by parallactic angle
                if derotate:
                    stack = derotate_frames(stack=stack,
                                            parang=parang,
                                            mask=roi_mask)

                # Merge into a single frame
                for merge in ('mean', 'median'):

                    # Merge by either taking the mean or median
                    with np.errstate(all='ignore'):
                        if merge == 'mean':
                            merged = np.mean(stack, axis=0)
                        elif merge == 'median':
                            merged = np.median(stack, axis=0)
                        else:
                            raise ValueError()

                    # Apply ROI mask
                    merged[~roi_mask] = np.nan

                    # Compute the figures of merit
                    figures_of_merit = \
                        compute_figures_of_merit(frame=merged,
                                                 position=position,
                                                 aperture_size=aperture_size,
                                                 ignore_neighbors=ignore,
                                                 optimize=optimize)

                    # Rename a few keys for FITS compatibility
                    figures_of_merit['noise'] = \
                        figures_of_merit.pop('noise_level')
                    figures_of_merit['old_pos'] = \
                        figures_of_merit.pop('old_position')
                    figures_of_merit['new_pos'] = \
                        figures_of_merit.pop('new_position')

                    # Save the result as a FITS file
                    file_name = (f'merged_{name}__'
                                 f'presub={presub}_'
                                 f'derotate={str(derotate)}_'
                                 f'merge={merge}.fits')
                    file_path = os.path.join(results_dir, file_name)
                    save_fits(array=merged,
                              file_path=file_path,
                              header=figures_of_merit)
                    
                    print(f'Saved {file_name}!')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
