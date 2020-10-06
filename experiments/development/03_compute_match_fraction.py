"""
Run consistency checks on best models and compute match fraction.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import os
import time

from astropy.units import Quantity

import matplotlib.pyplot as plt
import numpy as np

from hsr4hci.utils.config import load_config
from hsr4hci.utils.consistency_checks import get_match_fraction
from hsr4hci.utils.data import load_data
from hsr4hci.utils.fits import save_fits
from hsr4hci.utils.hdf import load_dict_from_hdf
from hsr4hci.utils.masking import get_roi_mask
from hsr4hci.utils.plotting import disable_ticks
from hsr4hci.utils.psf import get_psf_diameter
from hsr4hci.utils.units import set_units_for_instrument


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nASSEMBLE RESIDUALS AND COMPUTE SIGNAL ESTIMATE\n', flush=True)

    # -------------------------------------------------------------------------
    # Load experiment configuration and data
    # -------------------------------------------------------------------------

    # Define paths for experiment folder and results folder
    experiment_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    results_dir = experiment_dir / 'results'

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    file_path = experiment_dir / 'config.json'
    config = load_config(file_path=file_path.as_posix())
    print('Done!', flush=True)

    # Load frames, parallactic angles, etc. from HDF file
    print('Loading data set...', end=' ', flush=True)
    stack, parang, psf_template, observing_conditions, metadata = load_data(
        **config['dataset']
    )
    print('Done!', flush=True)

    # Load results HDF file
    print('Loading main results file...', end=' ', flush=True)
    file_path = results_dir / 'results.hdf'
    results = load_dict_from_hdf(file_path=file_path.as_posix())
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Define various useful shortcuts
    # -------------------------------------------------------------------------

    # Quantities related to the size of the data set
    n_frames, x_size, y_size = stack.shape
    frame_size = (x_size, y_size)
    center = (x_size / 2, y_size / 2)

    # Metadata of the data set
    pixscale = float(metadata['PIXSCALE'])
    lambda_over_d = float(metadata['LAMBDA_OVER_D'])

    # Experiment configuration
    n_test_positions = config['consistency_checks']['n_test_positions']
    filter_size = config['consistency_checks']['filter_size']

    # -------------------------------------------------------------------------
    # Activate the unit conversions for this instrument
    # -------------------------------------------------------------------------

    set_units_for_instrument(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
        verbose=False,
    )

    # -------------------------------------------------------------------------
    # Fit the PSF template with a 2D Moffat function to estimate its diameter
    # -------------------------------------------------------------------------

    print('Fitting PSF diameter...', end=' ', flush=True)
    psf_diameter = get_psf_diameter(
        psf_template=psf_template,
        pixscale=pixscale,
        lambda_over_d=lambda_over_d,
    )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Set up the mask for the region of interest (ROI)
    # -------------------------------------------------------------------------

    # Define a mask for the ROI
    roi_mask = get_roi_mask(
        mask_size=frame_size,
        inner_radius=Quantity(*config['roi_mask']['inner_radius']),
        outer_radius=Quantity(*config['roi_mask']['outer_radius']),
    )

    # -------------------------------------------------------------------------
    # Compute the consistency checks and get the match fraction for each pixel
    # -------------------------------------------------------------------------

    print('\nComputing match fraction:', flush=True)
    match_fraction = get_match_fraction(
        results=results,
        parang=parang,
        psf_diameter=psf_diameter,
        roi_mask=roi_mask,
        n_test_positions=n_test_positions,
    )
    print('')

    # -------------------------------------------------------------------------
    # Save match fraction as a FITS file
    # -------------------------------------------------------------------------

    print('Saving match fraction and masks to FITS...', end=' ', flush=True)
    file_path = results_dir / 'match_fraction.fits'
    save_fits(array=match_fraction, file_path=file_path.as_posix())
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create a PDF plot of the match fraction
    # -------------------------------------------------------------------------

    # Initialize plot directory
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Initialize grid for pcolormesh()
    x_range = np.arange(match_fraction.shape[0])
    y_range = np.arange(match_fraction.shape[1])
    x, y = np.meshgrid(x_range, y_range)

    # Plot the match fraction
    # Using pcolormesh() instead of imshow() avoids interpolation artifacts in
    # most PDF viewers (otherwise, the PDF version will often look blurry).
    plt.pcolormesh(
        x,
        y,
        match_fraction,
        vmin=-0,
        vmax=1,
        shading='nearest',
        cmap='viridis',
        snap=True,
        rasterized=True,
    )

    # Disable ticks and set plot size
    disable_ticks(plt.gca())
    plt.gcf().set_size_inches(4, 4, forward=True)

    # Save the result as a PDF
    file_path = plots_dir / 'match_fraction.pdf'
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0, dpi=600)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
