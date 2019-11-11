"""
Merge residuals that were generated on the cluster in parallel.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import time

from tqdm import tqdm

import numpy as np

from hsr4hci.utils.config import load_config
from hsr4hci.utils.fits import save_fits


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nMERGE RESIDUALS\n', flush=True)

    # Get expected size of the residual stack
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    config = load_config(os.path.join(experiment_dir, 'config.json'))
    frame_size = tuple(config['dataset']['frame_size'])

    # -------------------------------------------------------------------------
    # Collect all numpy files with residuals
    # -------------------------------------------------------------------------

    # Define the directory containing all the residuals
    residuals_dir = os.path.join(experiment_dir, 'results', 'positions',
                                 'residuals')

    # Get all numpy files in the residuals directory
    print('Collecting files in residuals directory...', end=' ', flush=True)
    numpy_files = [_ for _ in os.listdir(residuals_dir) if _.endswith('npy')]
    print(f'Done! ({len(numpy_files)} files found)\n', flush=True)

    # Collection detection maps that we read in
    residual_stack = None

    # Loop over all files, read them in and add residuals to the residual_stack
    print('Reading in numpy files from results directory:')
    for numpy_file in tqdm(numpy_files, ncols=80):

        # Get the position that the current file corresponds to
        position = \
            tuple(map(int, numpy_file.split('__')[1].split('.')[0].split('_')))

        # Read in the numpy file
        file_path = os.path.join(residuals_dir, numpy_file)
        array = np.load(file=file_path)

        # Once we have read in the first array, we finally know the shape of
        # the full residual stack and can properly initialize it
        if residual_stack is None:
            stack_shape = (len(array), ) + frame_size
            residual_stack = np.full(stack_shape, np.nan)

        # Add the array containing the residuals for the current position into
        # the residual_stack (at that position)
        residual_stack[:, position[0], position[1]] = array

    print('')

    # -------------------------------------------------------------------------
    # Merge residuals and save results
    # -------------------------------------------------------------------------

    results_dir = os.path.join(experiment_dir, 'results')

    print('Saving residual stack...', end=' ', flush=True)
    file_path = os.path.join(results_dir, 'residual_stack.fits')
    save_fits(residual_stack, file_path)
    print('Done!', flush=True)

    print('Averaging residual stack...', end=' ', flush=True)
    average_residuals = np.nanmedian(residual_stack, axis=0)
    print('Done!', flush=True)

    print('Saving average residuals...', end=' ', flush=True)
    file_path = os.path.join(results_dir, 'average_residuals.fits')
    save_fits(average_residuals, file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
