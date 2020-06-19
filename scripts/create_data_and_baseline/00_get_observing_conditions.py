"""
This script can be used to extract additional meta information about
the observing conditions (such as the coherence time, air mass, seeing,
etc.) directly from the raw FITS files of the ESO archive.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import json
import os
import re
import time

from tqdm import tqdm

import h5py
import numpy as np

from hsr4hci.utils.argparsing import get_base_directory
from hsr4hci.utils.fits import get_fits_header_value, \
    get_fits_header_value_array, header_value_exists
from hsr4hci.utils.observing_conditions import get_key_map

# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nGET OBSERVING CONDITIONS FROM RAW FITS FILES\n', flush=True)

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

    # Select meta data (about the data set) from the config file, and convert
    # the observation date to a proper datetime object (assuming UTC time)
    metadata = config['metadata']
    obs_date = datetime.fromisoformat(metadata['DATE'])
    obs_date = obs_date.replace(tzinfo=timezone.utc)

    # -------------------------------------------------------------------------
    # Get indices of selected frames from PynPoint database
    # -------------------------------------------------------------------------

    print('Collecting indices from PynPoint database...', end=' ', flush=True)

    # Load path to the PynPoint database and key to selected indices
    file_name = config['input_data']['stack']['file_name']
    file_path = os.path.join(base_dir, 'input', file_name)
    index_key = config['input_data']['stack']['index_key']

    # Read the indices of the selected frames, that is, the indices of the
    # frames in the raw FITS files that have passed quality control and are
    # also present in the pre-processed version of the data from PynPoint.
    # If no index key was given (e.g., for SPHERE data, where sometimes no
    # frame selection is performed), set the indices themselves also to None.
    if index_key is not None:
        with h5py.File(file_path, 'r') as hdf_file:
            indices = np.array(hdf_file[index_key])
            len_indices: Optional[int] = len(indices)
    else:
        indices = None
        len_indices = None

    print(f'Done! [len(indices) = {len_indices}]', flush=True)

    # -------------------------------------------------------------------------
    # Define paths and prepare results dictionary
    # -------------------------------------------------------------------------

    print('Collecting paths to FITS files...', end=' ', flush=True)

    # Get path to directory that contains raw FITS files, as well as any
    # potential patterns that the names of the FITS files need to match
    fits_dir = config['raw_data']['fits_dir']
    name_pattern = config['raw_data']['name_pattern']

    # Collect all FITS fits in the given FITS directory
    fits_files = \
        list(filter(lambda _: _.endswith('fits'), os.listdir(fits_dir)))

    # Filter out all files that do not match the given name pattern
    if name_pattern is not None:
        regex = re.compile(name_pattern)
        fits_files = list(filter(regex.search, fits_files))

    # Add base directory to file names
    fits_files = [os.path.join(fits_dir, _) for _ in fits_files]

    # Remove all files that do not contain an attribute for the coherence time
    # TODO: Check if there is a better way of filtering out these files!
    tau0_key = 'ESO TEL AMBI TAU0'
    fits_files = \
        list(filter(lambda _: header_value_exists(_, tau0_key), fits_files))

    # Read out the observation date from each FITS file and make sure the
    # list of FITS files are sorted by this date
    fits_files_with_dates = \
        [(_, get_fits_header_value(_, key='DATE-OBS', dtype=datetime))
         for _ in fits_files]
    fits_files_with_dates = sorted(fits_files_with_dates, key=lambda _: _[1])
    fits_files = [path for path, date in fits_files_with_dates]

    # Count the total number of files and frames (before frame selection)
    n_files = len(fits_files)
    n_frames = sum(get_fits_header_value(_, 'NAXIS3', int) for _ in fits_files)

    print(f'Done! [n_files = {n_files}, n_frames = {n_frames}]', flush=True)
    print('Preparing results dictionary...', end=' ', flush=True)

    # Initialize the dictionary of lists which will store the results
    results: Dict[str, list] = \
        {key: list() for key in get_key_map(obs_date=obs_date).keys()}

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Loop over raw FITS files (cubes) and read out meta-information
    # -------------------------------------------------------------------------

    print('Processing raw FITS files:', flush=True)

    warnings = list()
    for fits_file in tqdm(fits_files, ncols=80):

        # Loop over all parameters describing the observing conditions that we
        # expect from a data set with the given date and try to read them from
        # the header or the current FITS file
        for key, value in get_key_map(obs_date=obs_date).items():

            # Try to extract the current parameter from the FITS header
            try:
                parameter_values = \
                    get_fits_header_value_array(file_path=fits_file,
                                                start_key=value['start_key'],
                                                end_key=value['end_key'])
                results[key].append(parameter_values)

            # In case this parameter cannot be found, store a warning
            except KeyError:
                warnings.append(f'Expected parameter {key} not found!')

    # If there were warnings about missing keys, print them now
    if warnings:
        print('\nCAUTION! The following keys could not be found:')
        for warning in sorted(list(set(warnings))):
            print('--', warning)
        print('')

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

            # Skip expected keys that could not be found in FITS file
            if not results[key]:
                continue

            # Merge list of result arrays into single array
            values = np.hstack(results[key])

            # If we previously have loaded the indices from frame selection,
            # apply them now to select the correct values
            if indices is not None:
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
