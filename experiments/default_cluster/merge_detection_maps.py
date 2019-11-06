"""
Merge detection maps that were generated on the cluster in parallel.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import time

from tqdm import tqdm

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
    print('\nMERGE DETECTION MAPS\n', flush=True)

    # -------------------------------------------------------------------------
    # Collect all FITS file with detection maps
    # -------------------------------------------------------------------------

    # Get results directory with detection maps
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    detection_maps_dir = os.path.join(experiment_dir, 'results', 'positions',
                                      'detection_maps')

    # Get all FITS files in the results directory
    print('Collecting FITS files in results directory...', end=' ', flush=True)
    fits_files = \
        [_ for _ in os.listdir(detection_maps_dir) if _.endswith('fits')]
    print(f'Done! {len(fits_files)}\n', flush=True)

    # Collection detection maps that we read in
    detection_maps = list()

    # Collect detection maps
    print('Reading in FITS files from results directory:')
    for fits_file in tqdm(fits_files, ncols=80):
        file_path = os.path.join(detection_maps_dir, fits_file)
        detection_maps.append(read_fits(file_path))
    print('')

    # -------------------------------------------------------------------------
    # Merge detection maps and save results
    # -------------------------------------------------------------------------

    print('Merging detection maps...', end=' ', flush=True)
    detection_map = np.nansum(detection_maps, axis=0)
    print('Done!', flush=True)

    print('Saving combined detection map...', end=' ', flush=True)
    file_path = os.path.join(experiment_dir, 'results', 'detection_map.fits')
    save_fits(detection_map, file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
