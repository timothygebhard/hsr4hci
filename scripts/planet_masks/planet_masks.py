"""
Create binary masks for each data set which indicate the path of the
planets in the data.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import json
import os
import time

from astropy import units
from astropy.convolution import AiryDisk2DKernel

import numpy as np

from hsr4hci.utils.data import load_default_data
from hsr4hci.utils.derotating import derotate_combine
from hsr4hci.utils.fits import save_fits
from hsr4hci.utils.general import rotate_position
from hsr4hci.utils.forward_modeling import crop_psf_template, get_signal_stack


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCREATE PLANET MASKS\n', flush=True)

    # -------------------------------------------------------------------------
    # Loop over different data sets
    # -------------------------------------------------------------------------

    # Ensure the results directory exists
    results_dir = os.path.join('.', 'results')
    Path(results_dir).mkdir(exist_ok=True)

    # Loop over all data sets and process them
    for dataset in [
        '51_eridani__k1',
        'beta_pictoris__lp',
        'beta_pictoris__mp',
        'hip_65426__lp',
        'hr_8799__j',
        'hr_8799__lp',
        'pz_telescopii__lp',
        'r_cra__lp',
    ]:

        print(f'Processing data set: {dataset}')

        # Construct path to data set configuration file
        print('-- Loading configuration file...', end=' ', flush=True)
        file_path = os.path.join('../../datasets', f'{dataset}.json')
        with open(file_path, 'r') as json_file:
            config = json.load(json_file)
        print('Done!', flush=True)

        # Define shortcuts
        target_name = config['metadata']['TARGET_STAR'].replace(' ', '_')
        filter_name = config['metadata']['FILTER'].replace('\'', 'p')
        pixscale = config['metadata']['PIXSCALE']
        lambda_over_d = config['metadata']['LAMBDA_OVER_D']
        x_size, y_size = config['frame_size']
        center = (x_size / 2, y_size / 2)

        # Load stack, parallactic angles and PSF template
        print('-- Loading data...', end=' ', flush=True)
        stack, parang, psf_template, observing_conditions, metadata = \
            load_default_data(
                planet=f'{target_name}__{filter_name}', stacking_factor=1,
            )
        print(f'Done! [stack.shape = {stack.shape}]', flush=True)

        # In case there is no PSF template present, we need to create a fake
        # one using an Airy kernel of the appropriate size
        if psf_template.shape == (0, 0):

            # Create a 2D Airy disk of the correct size as a numpy array
            print('-- Creating dummy PSF template...', end=' ', flush=True)
            psf_template = AiryDisk2DKernel(
                radius=1.383 * lambda_over_d / pixscale,
                x_size=x_size,
                y_size=y_size,
            ).array
            print('Done!', flush=True)

        # Compute the cropping radius for the PSF
        psf_radius = units.Quantity(lambda_over_d / pixscale, 'pixel')
        print(f'-- Target PSF radius (in pixels): {psf_radius:.2f}')

        # Crop the PSF to a more appropriate size
        print('-- Cropping PSF template...', end=' ', flush=True)
        psf_cropped = crop_psf_template(
            psf_template=psf_template,
            psf_radius=psf_radius,
        )
        print('Done!', flush=True)

        # Instantiate the empty forward model
        forward_model = np.zeros_like(stack)

        # Loop over the different planets for the data set
        print('-- Computing forward models...', end=' ', flush=True)
        for key, values in config['evaluation']['planets'].items():

            # Get final position of the planet
            final_position = values['position'][::-1]

            starting_position = rotate_position(
                position=final_position,
                center=center,
                angle=parang[0],
            )

            # Compute a forward model for this planet
            planet_model, planet_positions = get_signal_stack(
                position=starting_position,
                frame_size=(x_size, y_size),
                parang=parang,
                psf_cropped=psf_cropped,
            )

            # Add this to the existing forward model
            forward_model += planet_model
        print('Done!', flush=True)

        # Create binary mask and save as as a FITS file
        print('-- Creating and saving binary mask...', end=' ', flush=True)
        binary_mask = (np.max(forward_model, axis=0) > 5e-1).astype(int)
        file_path = os.path.join(results_dir, f'{dataset}__binary_mask.fits')
        save_fits(binary_mask, file_path)
        print('Done!', flush=True)

        # Save result as a FITS file
        print('-- Creating and saving sanity check...', end=' ', flush=True)
        derotated = derotate_combine(
            stack=forward_model,
            parang=parang,
        )
        file_path = os.path.join(results_dir, f'{dataset}__sanity_check.fits')
        save_fits(derotated, file_path)
        print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'This took {time.time() - script_start:.1f} seconds!\n')
