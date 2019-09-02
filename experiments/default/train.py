"""
Train an HSR model, make predictions and derotate frames.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import time

from hsr4hci.utils.config import load_config
from hsr4hci.utils.data import load_data
from hsr4hci.utils.fits import save_fits
from hsr4hci.models.hsr import HalfSiblingRegression

from pathlib import Path

import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    @profile
    def run_main():
        # -------------------------------------------------------------------------
        # Preliminaries
        # -------------------------------------------------------------------------

        script_start = time.time()
        print('\nTRAIN PIXEL-WISE MODELS\n', flush=True)

        # -------------------------------------------------------------------------
        # Load config and data
        # -------------------------------------------------------------------------

        # Load experiment config from JSON
        config = load_config('./config.json')

        # Load frames and parallactic angles from HDF file
        stack, parang, psf_template = load_data(dataset_config=config['dataset'])

        # -------------------------------------------------------------------------
        # Set up and train model
        # -------------------------------------------------------------------------

        # Instantiate and train model
        print(f'Training HSR model (n_frames={len(stack)}):', flush=True)
        hsr = HalfSiblingRegression(config=config)
        hsr.train(stack=stack,
                  parang=parang,
                  psf_template=psf_template)
        print('', flush=True)

        # Get the detection map
        print(f'Computing detection map...', end=' ', flush=True)
        detection_map = hsr.get_detection_map()
        print('detection_map', detection_map.shape)
        print('Done!', flush=True)

        # -------------------------------------------------------------------------
        # Save results to HDF and FITS
        # -------------------------------------------------------------------------

        # Ensure the results dir exists
        results_dir = os.path.join(config['experiment_dir'], 'results')
        Path(results_dir).mkdir(exist_ok=True)

        # Store the detection map to a FITS file
        print(f'Saving results to FITS...', end=' ', flush=True)
        fits_file_path = os.path.join(results_dir, f'detection_map.fits')
        save_fits(array=detection_map, file_path=fits_file_path)
        print('Done!', flush=True)

        # -------------------------------------------------------------------------
        # Postliminaries
        # -------------------------------------------------------------------------

        print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')

    run_main()
