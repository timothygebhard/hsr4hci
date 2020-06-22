"""
This script essentially takes a PynPoint data base file (PynPoint is
used for the pre-processing of the raw data) and creates several HDF
files from it that contain the data at different levels of pre-stacking
(i.e., combining blocks of frames using the mean or median).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import json
import os
import time
import warnings

import bottleneck as bn
import h5py
import numpy as np

from hsr4hci.utils.argparsing import get_base_directory
from hsr4hci.utils.fits import read_fits
from hsr4hci.utils.general import prestack_array, get_md5_checksum, \
    is_fits_file, is_hdf_file, crop_center
from hsr4hci.utils.observing_conditions import load_observing_conditions


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

    # Get metadata for the data set
    metadata = config['metadata']

    # Get processing options
    frame_size = config['frame_size']
    stacking_factors = config['stacking_factors']

    # -------------------------------------------------------------------------
    # Load the stack, parallactic angles and PSF template (from HDF or FITS)
    # -------------------------------------------------------------------------

    print('Loading stack...', end=' ', flush=True)

    # Construct path to stack file
    stack_file_name = config['input_data']['stack']['file_name']
    stack_file_path = os.path.join(base_dir, 'input', stack_file_name)

    # Actually load the stack (from FITS or HDF)
    if is_fits_file(file_path=stack_file_path):
        stack = read_fits(file_path=stack_file_path)
    elif is_hdf_file(file_path=stack_file_path):
        stack_key = config['input_data']['stack']['stack_key']
        with h5py.File(stack_file_path, 'r') as hdf_file:
            stack = np.array(hdf_file[stack_key])
    else:
        raise RuntimeError('stack file must be either HDF or FITS!')

    # Crop the stack around the center to the desired spatial size
    stack = crop_center(stack, (-1, frame_size[0], frame_size[1]))

    print('Done!', flush=True)
    print('Loading parallactic angles...', end=' ', flush=True)

    # Construct path to parallactic angles file
    parang_file_name = config['input_data']['parang']['file_name']
    parang_file_path = os.path.join(base_dir, 'input', parang_file_name)

    # Actually load the parang (from FITS or HDF)
    if is_fits_file(file_path=parang_file_path):
        parang = read_fits(file_path=parang_file_path)
    elif is_hdf_file(file_path=parang_file_path):
        parang_key = config['input_data']['parang']['key']
        with h5py.File(parang_file_path, 'r') as hdf_file:
            parang = np.array(hdf_file[parang_key])
    else:
        raise RuntimeError('parang file must be either HDF or FITS!')

    print('Done!', flush=True)
    print('Loading PSF template...', end=' ', flush=True)

    # Load the PSF template, if there exists one. Since not all data sets have
    # a PSF template, this step is different from the stack and parang.
    file_name = config['input_data']['psf_template']['file_name']
    if file_name is not None:

        # Construct full path to PSF template file
        psf_file_path = os.path.join(base_dir, 'input', file_name)

        # Actually load the PSF template (from FITS or HDF)
        if is_fits_file(file_path=psf_file_path):
            psf_template = read_fits(file_path=psf_file_path)
        elif is_hdf_file(file_path=psf_file_path):
            psf_template_key = config['input_data']['psf_template']['key']
            with h5py.File(psf_file_path, 'r') as hdf_file:
                psf_template = np.array(hdf_file[psf_template_key])
        else:
            raise RuntimeError('psf_template file must be either HDF or FITS!')

        print('Done!', flush=True)

    # If no psf_template is given, add an empty array and raise a warning
    else:

        psf_template = np.empty((0, 0))
        print('Done!', flush=True)
        warnings.warn('No unsaturated PSF template given!')

    # -------------------------------------------------------------------------
    # Augment meta data based on the original stack file
    # -------------------------------------------------------------------------

    # Augment the metadata based on the "raw" stack file. This information is
    # intended to make it easier to trace back where the data in the resulting
    # HDF file originally came from.
    print('Augmenting metadata for data set...', end=' ', flush=True)
    metadata['ORIGINAL_STACK_FILE_NAME'] = stack_file_name
    metadata['ORIGINAL_STACK_FILE_HASH'] = get_md5_checksum(stack_file_path)
    metadata['ORIGINAL_STACK_SHAPE'] = stack.shape
    print('Done!')

    # -------------------------------------------------------------------------
    # Load observing conditions
    # -------------------------------------------------------------------------

    # Define expected file path for observing conditions
    oc_file_name = 'observing_conditions.hdf'
    oc_file_path = os.path.join(base_dir, 'observing_conditions', oc_file_name)

    # Load the observing conditions from the HDF file (as a dictionary)
    observing_conditions = \
        load_observing_conditions(file_path=oc_file_path,
                                  transform_wind_direction=False)

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

        # Created stacked versions of the observing condition parameters
        stacked_observing_conditions = dict()
        if observing_conditions is not None:
            for key, parameter in observing_conditions.items():
                stacked_observing_conditions[key] = \
                    prestack_array(array=parameter,
                                   stacking_factor=stacking_factor,
                                   stacking_function=bn.nanmedian)

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

            # Create new group for observing conditions and save them as well
            if stacked_observing_conditions:
                oc_group = hdf_file.create_group(name='observing_conditions')
                for key, parameter in stacked_observing_conditions.items():
                    oc_group.create_dataset(name=key,
                                            data=parameter,
                                            dtype=np.float32)

            # Add meta data about the data set and the instrument
            for key, value in metadata.items():
                hdf_file.attrs[key] = value

        print('Done!')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
