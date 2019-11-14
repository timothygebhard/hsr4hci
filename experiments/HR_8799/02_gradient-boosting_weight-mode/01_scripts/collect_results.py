"""
Collect results (residuals) from experiments and rename FITS files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import os
import shutil
import time


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCOLLECT EXPERIMENT RESULTS\n', flush=True)

    # -------------------------------------------------------------------------
    # Collect and copy experiment results
    # -------------------------------------------------------------------------

    # Make sure the folder for collected results exists
    collected_results_dir = '../03_results/collected_results'
    Path(collected_results_dir).mkdir(exist_ok=True, parents=True)

    # Read in the list of experiments
    with open('../02_experiments/list_of_experiments.txt', 'r') as text_file:
        list_of_experiments = [_.strip() for _ in text_file.readlines()]

    # Loop over all experiments and process them separately
    for experiment_name in list_of_experiments:

        # Copy results
        source_file = os.path.join('..', '02_experiments', experiment_name,
                                   'results', 'average_residuals.fits')
        target_file = os.path.join(collected_results_dir,
                                   f'{experiment_name}.fits')
        shutil.copy(source_file, target_file)
        print(f'Copied {source_file} -> {target_file}')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
