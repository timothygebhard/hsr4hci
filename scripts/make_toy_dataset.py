"""
Create toy data sets with fake planets.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import time

from astropy.units import Quantity

import h5py
import numpy as np

from hsr4hci.utils.config import get_data_dir
from hsr4hci.utils.data import load_default_data
from hsr4hci.utils.fits import save_fits
from hsr4hci.utils.forward_modeling import get_signal_stack
from hsr4hci.utils.psf import crop_psf_template
from hsr4hci.utils.units import set_units_for_instrument


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nMAKE TOY DATASET\n')

    # Define planet and wavelength of base data
    target_name = 'Beta_Pictoris'
    filter_name = 'Lp'

    # Define name of the resulting toy data set
    toy_name = 'dummy_1'

    # Define positions and amplitudes of fake planets
    positions_and_amplitudes = [
        ((42, 23), 60),
        ((25, 55), 20)
    ]

    # -------------------------------------------------------------------------
    # Load data on which the toy data set will be based
    # -------------------------------------------------------------------------

    # Load a data set
    print(f'Loading {target_name} ({filter_name})...', end=' ', flush=True)
    base_stack, parang, psf_template, obscon, metadata = load_default_data(
        planet='Beta_Pictoris__Lp', stacking_factor=50,
    )
    print('Done!', flush=True)

    # Remove offset from parallactic angles
    parang -= parang[0]

    # Enable instrument-specific unit conversions
    set_units_for_instrument(
        pixscale=Quantity(metadata['PIXSCALE'], 'arcsecond / pixel'),
        lambda_over_d=Quantity(metadata['LAMBDA_OVER_D'], 'arcsecond'),
    )

    # Crop the PSF template to desired size
    print(f'Cropping PSF template...', end=' ', flush=True)
    psf_cropped = crop_psf_template(
        psf_template=psf_template, psf_radius=Quantity(3, 'lambda_over_d')
    )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create fake planets and add them on top of the "noise"
    # -------------------------------------------------------------------------

    print('Creating fake planets...', end=' ', flush=True)

    # Define dummy stack to keep adding up the pure signal
    signal = np.zeros(base_stack.shape)

    planet_positions = []

    # Loop over positions and amplitudes and compute forward models
    for position, amplitude in positions_and_amplitudes:

        # Generate the forward model (i.e., the fake planet signal)
        tmp_signal, planet_positions_ = get_signal_stack(
            position=position,
            frame_size=base_stack.shape[1:],
            parang=parang,
            psf_cropped=psf_cropped
        )
        planet_positions.append(np.array(planet_positions_))

        # Scale to desired amplitude
        tmp_signal /= np.max(tmp_signal)
        tmp_signal *= amplitude

        # Add planet to the signal stack
        signal += tmp_signal

    planet_positions = np.array(planet_positions)

    print('Done!', flush=True)
    print('Adding signal to noise...', end=' ', flush=True)

    # Add simulated signals on top of the "noise" (i.e., the base data stack)
    stack = base_stack + signal

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Save results to an HDF file (and also as a FITS file)
    # -------------------------------------------------------------------------

    print('Saving toy dataset...', end=' ', flush=True)

    # Prepare output directory
    data_dir = get_data_dir()
    toy_data_dir = data_dir / 'toy_datasets'
    results_dir = toy_data_dir / 'Beta_Pictoris' / toy_name
    results_dir.mkdir(exist_ok=True, parents=True)

    # Save planet positions to a *.npy file
    file_path = results_dir / 'positions.npy'
    np.save(file_path.as_posix(), arr=planet_positions)

    # Save the parallactic angles to a *.npy file
    file_path = results_dir / 'parang.npy'
    np.save(file=file_path.as_posix(), arr=parang)

    # Save main results as a HDF file
    file_path = results_dir / 'data.hdf'
    with h5py.File(file_path, 'w') as hdf_file:
        hdf_file.create_dataset(name='stack', data=stack)
        hdf_file.create_dataset(name='parang', data=parang)
        hdf_file.create_dataset(name='psf_template', data=psf_template)
        for key, value in metadata.items():
            hdf_file.attrs[key] = value

    # Create a FITS compatible version of the metadata dict that can be
    # written to the FITS header (i.e., prepend HIERARCH to all keywords)
    header = {'HIERARCH ' + key: value for key, value in metadata.items()}

    # Save results as a FITS file
    file_path = results_dir / 'data.fits'
    save_fits(stack, file_path=file_path, header=header)

    # Save pure signal as a FITS file
    file_path = results_dir / 'signal.fits'
    save_fits(signal, file_path=file_path)

    print('Done!\n', flush=True)
