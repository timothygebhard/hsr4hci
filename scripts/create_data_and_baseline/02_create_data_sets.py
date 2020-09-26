"""
This script essentially takes a PynPoint data base file (PynPoint is
used for the pre-processing of the raw data) and creates several HDF
and FITS files from it that contain the data at different levels of
pre-stacking (i.e., combining blocks of frames using the mean/median).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import json
import time
import warnings

from astropy.units import Quantity

import bottleneck as bn
import h5py
import numpy as np

from hsr4hci.utils.argparsing import get_base_directory
from hsr4hci.utils.fits import is_fits_file, read_fits, save_fits
from hsr4hci.utils.forward_modeling import get_planet_paths
from hsr4hci.utils.general import (
    crop_center,
    get_md5_checksum,
    prestack_array,
)
from hsr4hci.utils.hdf import is_hdf_file
from hsr4hci.utils.observing_conditions import load_observing_conditions
from hsr4hci.utils.psf import crop_psf_template, get_artificial_psf
from hsr4hci.utils.units import set_units_for_instrument


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

    # Read in the config file and parse it
    file_path = base_dir / 'config.json'
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)

    # -------------------------------------------------------------------------
    # Define shortcuts and set up unit conversions
    # -------------------------------------------------------------------------

    # Shortcuts to PIXSCALE and LAMBDA_OVER_D
    metadata = config['metadata']
    pixscale = metadata['PIXSCALE']
    lambda_over_d = metadata['LAMBDA_OVER_D']

    # Use this to set up the instrument-specific conversion factors. We need
    # this here to that we can parse "lambda_over_d" as a unit in the config.
    set_units_for_instrument(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
    )

    # Other shortcuts to elements in the config
    frame_size = config['frame_size']
    stacking_factors = config['stacking_factors']
    planet_config = config['evaluation']['planets']

    # -------------------------------------------------------------------------
    # Load the stack (from HDF or FITS)
    # -------------------------------------------------------------------------

    print('Loading stack...', end=' ', flush=True)

    # Construct path to stack file
    stack_file_name = config['input_data']['stack']['file_name']
    stack_file_path = base_dir / 'input' / stack_file_name

    # Actually load the stack (from FITS or HDF)
    if is_fits_file(stack_file_path):
        stack = read_fits(stack_file_path)
    elif is_hdf_file(stack_file_path):
        stack_key = config['input_data']['stack']['stack_key']
        with h5py.File(stack_file_path, 'r') as hdf_file:
            stack = np.array(hdf_file[stack_key])
    else:
        raise RuntimeError('stack file must be either HDF or FITS!')

    # Crop the stack around the center to the desired spatial size
    stack = crop_center(stack, (-1, frame_size[0], frame_size[1]))

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Load the parallactic angles (from HDF or FITS)
    # -------------------------------------------------------------------------

    print('Loading parallactic angles...', end=' ', flush=True)

    # Construct path to parallactic angles file
    parang_file_name = config['input_data']['parang']['file_name']
    parang_file_path = base_dir / 'input' / parang_file_name

    # Actually load the parang (from FITS or HDF)
    if is_fits_file(parang_file_path):
        parang = read_fits(parang_file_path)
    elif is_hdf_file(parang_file_path):
        parang_key = config['input_data']['parang']['key']
        with h5py.File(parang_file_path, 'r') as hdf_file:
            parang = np.array(hdf_file[parang_key])
    else:
        raise RuntimeError('parang file must be either HDF or FITS!')

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Load the unsaturated PSF template (from HDF or FITS)
    # -------------------------------------------------------------------------

    print('Loading PSF template...', end=' ', flush=True)

    # Load the PSF template, if there exists one. Since not all data sets have
    # a PSF template, this step is different from the stack and parang.
    file_name = config['input_data']['psf_template']['file_name']
    if file_name is not None:

        # Construct full path to PSF template file
        psf_file_path = base_dir / 'input' / file_name

        # Actually load the PSF template (from FITS or HDF)
        if is_fits_file(psf_file_path):
            psf_template: np.ndarray = read_fits(psf_file_path)
        elif is_hdf_file(psf_file_path):
            psf_template_key = config['input_data']['psf_template']['key']
            with h5py.File(psf_file_path, 'r') as hdf_file:
                psf_template = np.array(hdf_file[psf_template_key]).squeeze()
        else:
            raise RuntimeError('psf_template file must be either HDF or FITS!')

        # Make sure that the PSF template is actually 2D
        if psf_template.ndim == 3:
            psf_template = np.mean(psf_template, axis=0)

        print('Done!', flush=True)

    # If no psf_template is given, add an empty array and raise a warning
    else:

        psf_template = np.empty((0, 0))
        print('Done!', flush=True)
        warnings.warn('No unsaturated PSF template given!')

    # Create a cropped version of the PSF template. If necessary (i.e., if no
    # real template is available) use an artificial PSF template.
    if psf_template.shape != (0, 0):
        psf_cropped = crop_psf_template(
            psf_template=psf_template,
            psf_radius=Quantity(1, 'lambda_over_d'),
        )
    else:
        fake_psf_template = get_artificial_psf(
            pixscale=Quantity(pixscale, 'arcsec / pixel'),
            lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
        )
        psf_cropped = crop_psf_template(
            psf_template=fake_psf_template,
            psf_radius=Quantity(1, 'lambda_over_d'),
        )

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
    # Load observing conditions (from HDF)
    # -------------------------------------------------------------------------

    # Define expected file path for observing conditions
    oc_file_name = 'observing_conditions.hdf'
    oc_file_path = base_dir / 'observing_conditions' / oc_file_name

    # Load the observing conditions from the HDF file (as a dictionary)
    observing_conditions = load_observing_conditions(
        file_path=oc_file_path, transform_wind_direction=False
    )

    # -------------------------------------------------------------------------
    # Loop over stacking factors and create a stacked data set for each
    # -------------------------------------------------------------------------

    # Make sure the directory for the processed data sets exists
    output_dir = base_dir / 'processed'
    output_dir.mkdir(exist_ok=True)

    print('\nStacking frames and saving results:', flush=True)

    # Loop over all stacking factors and create data set
    for stacking_factor in stacking_factors:

        print(
            f'-- Running for stacking factor = {stacking_factor}...',
            end=' ',
            flush=True,
        )

        # ---------------------------------------------------------------------
        # Create pre-stacked version of stack, parang, obs_con and planet mask
        # ---------------------------------------------------------------------

        # Stack frames and parallactic angles (use median for frames and mean
        # for parallactic angles, as those are the PynPoint defaults)
        stacked_stack = prestack_array(
            array=stack,
            stacking_factor=stacking_factor,
            stacking_function=bn.nanmedian,
        )
        stacked_parang = prestack_array(
            array=parang,
            stacking_factor=stacking_factor,
            stacking_function=bn.nanmean,
        )

        # Created stacked versions of the observing condition parameters
        stacked_observing_conditions = dict()
        if observing_conditions is not None:
            for key, parameter in observing_conditions.items():
                stacked_observing_conditions[key] = prestack_array(
                    array=parameter,
                    stacking_factor=stacking_factor,
                    stacking_function=bn.nanmedian,
                )

        # Compute a binary mask of the pixels on the planet path(s) (i.e., the
        # spatial pixels that at some point in time contain planet signal), as
        # well as the exact positions of the planet(s) for each time step
        planet_paths_mask, planet_positions = get_planet_paths(
            stack_shape=stacked_stack.shape,
            parang=stacked_parang,
            psf_cropped=psf_cropped,
            planet_config=planet_config,
            threshold=5e-1,
        )

        # Cast the mask to integer, because FITS complains about binary masks
        planet_paths_mask = planet_paths_mask.astype(int)

        # ---------------------------------------------------------------------
        # Save everything to a new HDF file
        # ---------------------------------------------------------------------

        # Construct file path for output HDF file
        file_path = output_dir / f'stacked__{stacking_factor}.hdf'

        # Save the result as an HDF file in the output directory
        with h5py.File(file_path, 'w') as hdf_file:

            # Save data sets: stack, parallactic angles, PSF template, as well
            # as a binary mask indicating the paths of the planets in the data
            hdf_file.create_dataset(
                name='stack', data=stacked_stack, dtype=np.float32
            )
            hdf_file.create_dataset(
                name='parang', data=stacked_parang, dtype=np.float32
            )
            hdf_file.create_dataset(
                name='psf_template', data=psf_template, dtype=np.float32
            )
            hdf_file.create_dataset(
                name='planet_paths_mask',
                data=planet_paths_mask,
                dtype=np.int
            )

            # Create a group for the planet positions
            pp_group = hdf_file.create_group(name='planet_positions')
            for planet_name, positions in planet_positions.items():
                pp_group.create_dataset(
                    name=planet_name, data=positions, dtype=np.float32
                )

            # Create new group for observing conditions and save them as well
            if stacked_observing_conditions:
                oc_group = hdf_file.create_group(name='observing_conditions')
                for key, parameter in stacked_observing_conditions.items():
                    oc_group.create_dataset(
                        name=key, data=parameter, dtype=np.float32
                    )

            # Add meta data about the data set and the instrument
            for key, value in metadata.items():
                hdf_file.attrs[key] = value

        # Create a FITS compatible version of the metadata dict that can be
        # written to the FITS header (i.e., prepend HIERARCH to all keywords)
        header = {'HIERARCH ' + key: value for key, value in metadata.items()}

        # Save the stacked version of the stack (without parang or observing
        # conditions) as a FITS file for quick inspection
        file_path = output_dir / f'stacked__{stacking_factor}.fits'
        save_fits(array=stacked_stack, file_path=file_path, header=header)

        # Also save the planet_paths_mask to FITS for quick inspection
        file_path = output_dir / f'planet_paths_mask__{stacking_factor}.fits'
        save_fits(array=planet_paths_mask, file_path=file_path)

        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Finally, also save the PSF template as FITS for quick inspection
    # -------------------------------------------------------------------------

    print('\nSaving unsaturated PSF template to FITS...', end=' ', flush=True)

    # Construct file name for PSF template FITS file
    file_path = output_dir / 'psf_template.fits'

    # Save the PSF template as a FITS file
    save_fits(array=psf_template, file_path=file_path)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
