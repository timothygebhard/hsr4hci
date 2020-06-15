"""
This script can be used to extract additional meta information about
the observing conditions (such as the coherence time, air mass, seeing,
etc.) directly from the raw FITS files of the ESO archive.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict

import json
import os
import time

from tqdm import tqdm

import h5py
import numpy as np

from hsr4hci.utils.argparsing import get_base_directory
from hsr4hci.utils.fits import get_fits_header_value_array
from hsr4hci.utils.observing_conditions import get_key_map


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCOLLECT ADDITIONAL INFORMATION FROM FITS FILES\n', flush=True)

    # -------------------------------------------------------------------------
    # Parse command line arguments and load config.json
    # -------------------------------------------------------------------------

    # Get base_directory from command line arguments
    base_dir = get_base_directory()

    # Construct expected path to config.json
    file_path = os.path.join(base_dir, 'config.json')

    # Read in the config file and parse it
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)

    # Select meta data (about the data set) from the config file
    metadata = config['metadata']

    # -------------------------------------------------------------------------
    # Get indices of selected frames from PynPoint database
    # -------------------------------------------------------------------------

    print('Collecting indices from PynPoint database...', end=' ', flush=True)

    # Define path to the PynPoint database
    pynpoint_db_file_path = config['raw_data']['pynpoint_db_file_path']
    pynpoint_db_index_key = config['raw_data']['pynpoint_db_index_key']

    # Read the indices of the selected frames
    with h5py.File(pynpoint_db_file_path, 'r') as hdf_file:
        indices = np.array(hdf_file[pynpoint_db_index_key])

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Define paths and prepare results dictionary
    # -------------------------------------------------------------------------

    print('Collecting paths to FITS files...', end=' ', flush=True)

    # Define path to directory that contains raw FITS files
    fits_files_base_dir = config['raw_data']['fits_files_base_dir']

    # Construct a list of the paths of all FITS files in this directory
    fits_files = os.listdir(fits_files_base_dir)
    fits_files = [_ for _ in fits_files if _.endswith('fits')]
    fits_files = [os.path.join(fits_files_base_dir, _) for _ in fits_files]
    fits_files = sorted(fits_files)

    print('Done!', flush=True)
    print('Preparing results dictionary...', end=' ', flush=True)

    # Initialize the dictionary of lists which will store the results
    results: Dict[str, list] = {key: list() for key in get_key_map().keys()}

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Loop over raw FITS files (cubes) and read out meta-information
    # -------------------------------------------------------------------------

    print('Processing raw FITS files:', flush=True)
    for fits_file in tqdm(fits_files, ncols=80):

        # Loop over all parameters describing the observing conditions and
        # retrieve them from the FITS file
        for key, eso_keys in get_key_map().items():
            parameter_values = \
                get_fits_header_value_array(file_path=fits_file,
                                            start_key=eso_keys['start_key'],
                                            end_key=eso_keys['end_key'])
            results[key].append(parameter_values)

    # -------------------------------------------------------------------------
    # Merge results list into numpy arrays and save the result as HDF
    # -------------------------------------------------------------------------

    print('Saving results to HDF file...', end=' ', flush=True)

    # Create the results directory for the observing conditions
    results_dir = Path(base_dir) / 'observing_conditions'
    results_dir.mkdir(exist_ok=True)

    # Construct path to the result HDF file
    results_file_path = (results_dir / 'observing_conditions.hdf').as_posix()

    with h5py.File(results_file_path, 'w') as hdf_file:

        # Add meta data about the data set and the instrument
        for key, value in metadata.items():
            hdf_file.attrs[key] = value

        # Loop over different quantities
        for key in results.keys():

            # Merge list of result arrays into single array and apply
            # indices from frame selection
            values = np.hstack(results[key])
            values = values[indices]

            # Save them to the HDF file
            hdf_file.create_dataset(name=key,
                                    dtype=np.float32,
                                    data=values)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
