"""
Create plots of pixel coefficients.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import time

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from photutils import CircularAperture

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from hsr4hci.coordinates import get_center
from hsr4hci.data import load_psf_template
from hsr4hci.fits import read_fits
from hsr4hci.plotting import disable_ticks, add_colorbar_to_ax, get_cmap
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
    for dataset, binning_factor, (x, y) in (
        ('beta_pictoris__lp', 1, (8 - 1, 30 - 1)),
        ('beta_pictoris__lp', 1, (38 - 1, 10 - 1)),
        ('beta_pictoris__mp', 1, (42 - 1, 19 - 1)),
        ('beta_pictoris__mp', 1, (28 - 1, 40 - 1)),
        ('r_cra__lp', 1, (12 - 1, 17 - 1)),
        ('r_cra__lp', 1, (33 - 1, 45 - 1)),
    ):

        start_time = time.time()
        print(f'Running for {dataset} ({x}, {y})...', end=' ', flush=True)

        # Load PSF and fit FWHM
        psf_template = load_psf_template(name_or_path=dataset)
        psf_fwhm = get_psf_fwhm(psf_template)

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
        fig, ax = plt.subplots(figsize=(2.4, 4))
        img = ax.pcolormesh(
            np.arange(x_size),
            np.arange(y_size),
            frame,
            vmin=-1,
            vmax=1,
            shading='nearest',
            cmap=get_cmap('RdBu_r'),
            snap=True,
            rasterized=True,
        )
        disable_ticks(ax)
        ax.set_aspect('equal')

        # Plot markers for position and center
        plt.plot(x, y, 'x', mew=2, ms=5, color='lime')
        plt.plot(center[0], center[1], '+', color='black')

        # Plot circular aperture for opposite position
        aperture = CircularAperture(
            positions=(2 * center[0] - x, 2 * center[0] - y), r=psf_fwhm
        )
        # noinspection PyTypeChecker
        aperture.plot(ls=':', lw=0.5, color='black')

        # Create the scale bar and add it to the frame
        scalebar = AnchoredSizeBar(
            transform=ax.transData,
            size=0.5 / 0.0271,
            label='0.5"',
            loc=2,
            pad=1,
            color='black',
            frameon=False,
            size_vertical=0,
            fontproperties=fm.FontProperties(size=6),
        )
        ax.add_artist(scalebar)

        # Add a color bar
        cbar = add_colorbar_to_ax(img, fig, ax, where='bottom')
        cbar.set_ticks([-0.66, -0.33, 0, 0.33, 0.66])
        cbar.ax.set_xlabel('Normalized coefficient value', fontsize=6)
        cbar.ax.xaxis.set_label_position('bottom')
        cbar.ax.tick_params(labelsize=6)

        # Save plot as a PDF
        fig.tight_layout()
        file_path = plots_dir / f'{dataset}__{binning_factor}__{x}_{y}.pdf'
        plt.savefig(file_path, bbox_inches='tight', dpi=600, pad_inches=0.005)

        print(f'Done! ({time.time() - start_time:.1f} seconds)', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
