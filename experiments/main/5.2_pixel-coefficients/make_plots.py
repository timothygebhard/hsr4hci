"""
Create plots of pixel coefficients.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import time

from photutils import CircularAperture

import matplotlib.pyplot as plt
import numpy as np

from hsr4hci.coordinates import get_center
from hsr4hci.config import get_experiments_dir
from hsr4hci.data import (
    load_metadata,
    load_psf_template,
)
from hsr4hci.fits import read_fits
from hsr4hci.plotting import plot_frame
from hsr4hci.psf import get_psf_fwhm


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nPLOT PIXEL COEFFICIENTS\n', flush=True)

    # -------------------------------------------------------------------------
    # Parse command line arguments
    # -------------------------------------------------------------------------

    # Set up parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--binning-factor',
        type=int,
        default=1,
        help='Binning factor for which to create plots.',
    )
    args = parser.parse_args()
    binning_factor = str(args.binning_factor)

    # -------------------------------------------------------------------------
    # Plot pixel coefficients and save results as PDF
    # -------------------------------------------------------------------------

    # Ensure the plots directory exists
    plots_dir = (
        get_experiments_dir()
        / 'main'
        / '5.2_pixel-coefficients'
        / 'plots'
        / f'binning-factor_{binning_factor}'
    )
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Loop over different data sets and positions
    for dataset, (x, y), loc in (
        ('beta_pictoris__lp', (8 - 1, 30 - 1), 'upper right'),
        ('beta_pictoris__lp', (38 - 1, 10 - 1), 'upper right'),
        ('beta_pictoris__mp', (42 - 1, 19 - 1), 'upper right'),
        ('beta_pictoris__mp', (28 - 1, 40 - 1), 'upper right'),
        ('r_cra__lp', (26 - 1, 10 - 1), 'upper right'),
        ('r_cra__lp', (33 - 1, 45 - 1), 'upper left'),
    ):

        start_time = time.time()
        print(f'Running for {dataset} ({x}, {y})...', end=' ', flush=True)

        # Load PSF and fit FWHM
        psf_template = load_psf_template(name_or_path=dataset)
        psf_fwhm = get_psf_fwhm(psf_template)

        # Load metadata
        metadata = load_metadata(name_or_path=dataset)
        pixscale = metadata['PIXSCALE']

        # Load pixel coefficients from FITS
        fits_dir = (
            get_experiments_dir()
            / 'main'
            / '5.2_pixel-coefficients'
            / dataset
            / f'binning-factor_{binning_factor}'
            / 'fits'
        )
        file_path = fits_dir / 'coefficients.fits'
        coefficients = read_fits(file_path=file_path, return_header=False)

        # Select coefficients for plot (flip coordinates because numpy!).
        # This is a 3D array, where the first dimension corresponds to the
        # number of training splits.
        frames = coefficients[y, x]

        # Average frames along the axis of the training splits and normalize
        # such that the (absolute) maximum is 1
        frame = np.asarray(np.mean(frames, axis=0))
        frame /= np.nanmax(np.abs(frame))

        # Define shortcuts
        x_size, y_size = frame.shape
        frame_size = (x_size, y_size)
        center = get_center(frame_size)

        # Plot the frame with the coefficients
        fig, ax, cbar = plot_frame(
            frame=frame,
            positions=[],
            labels=[],
            pixscale=pixscale,
            figsize=(2.9 / 2.54, 3.5 / 2.54),
            subplots_adjust=dict(left=0.005, top=1, right=0.995, bottom=0.105),
            scalebar_color='black',
            limits=(-1, 1),
            scalebar_loc=loc,
        )

        # Plot markers for position and center
        ax.plot(x, y, 'x', mew=1 , ms=3, color='lime')
        CircularAperture(
            positions=(2 * center[0] - x, 2 * center[0] - y), r=psf_fwhm
        ).plot(axes=ax, **dict(ls=':', lw=1, color='black'))

        # Save plot as a PDF
        file_path = plots_dir / f'{dataset}__{x}_{y}.pdf'
        plt.savefig(file_path, dpi=600, pad_inches=0)

        print(f'Done! ({time.time() - start_time:.1f} seconds)', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
