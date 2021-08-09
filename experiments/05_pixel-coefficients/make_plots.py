"""
Create plots of pixel coefficients.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import time

from photutils import CircularAperture

import matplotlib.pyplot as plt

from hsr4hci.coordinates import get_center
from hsr4hci.data import load_psf_template, load_metadata
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
    # Plot pixel coefficients and save results as PDF
    # -------------------------------------------------------------------------

    # Ensure the plots directory exists
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    # Loop over different data sets and positions
    for dataset, binning_factor, (x, y), loc in (
        ('beta_pictoris__lp', 1, (8 - 1, 30 - 1), 1),
        ('beta_pictoris__lp', 1, (38 - 1, 10 - 1), 1),
        ('beta_pictoris__mp', 1, (42 - 1, 19 - 1), 1),
        ('beta_pictoris__mp', 1, (28 - 1, 40 - 1), 1),
        ('r_cra__lp', 1, (12 - 1, 17 - 1), 1),
        ('r_cra__lp', 1, (33 - 1, 45 - 1), 2),
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
        fits_dir = Path(dataset) / f'binning-factor_{binning_factor}' / 'fits'
        file_path = fits_dir / 'coefficients.fits'
        coefficients = read_fits(file_path=file_path, return_header=False)

        # Select coefficients for plot (flip coordinates because numpy!)
        frame = coefficients[y, x]

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
        file_path = plots_dir / f'{dataset}__{binning_factor}__{x}_{y}.pdf'
        plt.savefig(file_path, dpi=600, pad_inches=0)

        print(f'Done! ({time.time() - start_time:.1f} seconds)', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
