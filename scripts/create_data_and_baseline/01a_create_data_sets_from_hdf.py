"""
This script essentially takes a PynPoint data base file (PynPoint is
used for the pre-processing of the raw data) and creates several HDF
files from it that contain the data at different levels of pre-stacking
(i.e., combining blocks of frames using the mean or median).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from hashlib import md5
from pathlib import Path

import json
import os
import time
import warnings

import bottleneck as bn
import h5py
import numpy as np

from hsr4hci.utils.argparsing import get_base_directory
from hsr4hci.utils.data import load_data
from hsr4hci.utils.general import prestack_array


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_md5_checksum(
    file_path: str,
    buffer_size: int = 8192
) -> str:
    """
    Compute the MD5 checksum of the file at the given `file_path`.

    Args:
        file_path: The to the file of which we want to compute the MD5
            checksum.
        buffer_size: Buffer size (in bytes) for reading in the target
            file in chunks.

    Returns:
        The MD5 checksum of the specified file.
    """

    # Initialize the MD5 checksum
    md5_checksum = md5()

    # Open the input file and process it in chunks, updating the MD5 hash
    with open(file_path, 'rb') as binary_file:
        for chunk in iter(lambda: binary_file.read(buffer_size), b""):
            md5_checksum.update(chunk)

    return str(md5_checksum.hexdigest())


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCREATE DATA SETS\n', flush=True)

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

    # -------------------------------------------------------------------------
    # Define shortcuts to various parts of the config
    # -------------------------------------------------------------------------

    # Get name of the HDF file containing the raw data and construct path to
    # the expected location (inside a folder named "raw")
    hdf_file_name = config['hdf_file_name']
    hdf_file_path = os.path.join(base_dir, 'raw', hdf_file_name)

    # Get keys of the data sets within the HDF file that we need to load
    stack_key = config['stack_key']
    parang_key = config['parang_key']
    psf_template_key = config['psf_template_key']

    # Get metadata for the data set
    metadata = config['metadata']

    # Get processing options
    frame_size = config['frame_size']
    stacking_factors = config['stacking_factors']

    # -------------------------------------------------------------------------
    # Load data set and crop it to the desired spatial resolution
    # -------------------------------------------------------------------------

    # Load the data from the HDF file and crop to target frame_size
    print('Loading raw data from HDF file...', end=' ', flush=True)
    stack, parang, psf_template = load_data(file_path=hdf_file_path,
                                            stack_key=stack_key,
                                            parang_key=parang_key,
                                            psf_template_key=psf_template_key,
                                            frame_size=frame_size)
    print('Done!')

    # Augment the metadata based on the "raw" HDF file. This information is
    # intended to make it easier to trace back where the data in the resulting
    # HDF file originally came from.
    print('Augmenting metadata for data set...', end=' ', flush=True)
    metadata['ORIGINAL_FILE_NAME'] = hdf_file_name
    metadata['ORIGINAL_FILE_HASH'] = get_md5_checksum(file_path=hdf_file_path)
    metadata['ORIGINAL_STACK_SHAPE'] = stack.shape
    print('Done!')

    # -------------------------------------------------------------------------
    # Define a "default PSF template"
    # -------------------------------------------------------------------------

    # Adding an "empty" placeholder template feels better than adding a
    # "default" template from another data set, as this information (i.e.,
    # the template being from another data set) might easily get lost.
    if psf_template is None:
        psf_template = np.empty((0, 0))
        warnings.warn('No unsaturated PSF template given!')

    # -------------------------------------------------------------------------
    # Loop over stacking factors and create a stacked data set for each
    # -------------------------------------------------------------------------

    # Make sure the directory for the processed data sets exists
    output_dir = os.path.join(base_dir, 'processed')
    Path(output_dir).mkdir(exist_ok=True)

    print('\nStacking frames and saving results:', flush=True)

    # Loop over all stacking factors and create data set
    for stacking_factor in stacking_factors:

        print(f'-- Running for stacking factor = {stacking_factor}...',
              end=' ', flush=True)

        # Stack frames and parallactic angles (use median for frames and mean
        # for parallactic angles, as those are the PynPoint defaults)
        stacked_stack = prestack_array(array=stack,
                                       stacking_factor=stacking_factor,
                                       stacking_function=bn.nanmedian)
        stacked_parang = prestack_array(array=parang,
                                        stacking_factor=stacking_factor,
                                        stacking_function=bn.nanmean)

        # Construct file name for output HDF file
        file_name = os.path.join(output_dir, f'stacked_{stacking_factor}.hdf')

        # Save the result as an HDF file in the output directory
        with h5py.File(file_name, 'w') as hdf_file:

            # Save data sets: stack, parallactic angles and PSF template
            hdf_file.create_dataset(name='stack',
                                    data=stacked_stack,
                                    dtype=np.float32)
            hdf_file.create_dataset(name='parang',
                                    data=stacked_parang,
                                    dtype=np.float32)
            hdf_file.create_dataset(name='psf_template',
                                    data=psf_template,
                                    dtype=np.float32)

            # Add meta data about the data set and the instrument
            for key, value in metadata.items():
                hdf_file.attrs[key] = value

        print('Done!')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
