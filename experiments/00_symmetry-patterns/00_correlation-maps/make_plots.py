"""
Apply clustering to time series and plot the results.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import time

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from hsr4hci.coordinates import get_center
from hsr4hci.fits import read_fits
from hsr4hci.plotting import disable_ticks, add_colorbar_to_ax


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nPLOT CORRELATION MAPS\n', flush=True)

    # -------------------------------------------------------------------------
    # Run k-means clustering for two example data sets
    # -------------------------------------------------------------------------

    # Ensure that FITS directory exists
    fits_dir = Path('fits')
    if not fits_dir.exists():
        raise NotADirectoryError('FITS directory not found!')

    # Ensure the plots directory exists
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    # Loop over different data sets
    for dataset, (x, y) in (
        ('beta_pictoris__mp', (30 - 1, 26 - 1)),
        ('hr_8799__lp', (26 - 1, 13 - 1)),
    ):

        start_time = time.time()
        print(f'Running for {dataset}...', end=' ', flush=True)

        # Load correlation coefficients from FITS
        file_path = fits_dir / f'{dataset}.fits'
        correlations = read_fits(file_path=file_path, return_header=False)

        # Select CC map that we want to plot (flip coordinates because numpy!)
        cc_map = correlations[y, x]

        # Define shortcuts
        x_size, y_size = cc_map.shape
        frame_size = (x_size, y_size)
        center = get_center(frame_size)

        # Plot the correlation map
        fig, ax = plt.subplots(figsize=(5, 6))
        img = ax.pcolormesh(
            np.arange(x_size),
            np.arange(y_size),
            cc_map,
            vmin=-0.6,
            vmax=0.6,
            shading='nearest',
            cmap='RdBu_r',
            snap=True,
            rasterized=True,
        )
        disable_ticks(ax)
        ax.set_aspect('equal')

        # Plot markers for position and center
        plt.plot(x, y, 'x', mew=2, ms=8, color='lime')
        plt.plot(center[0], center[1], '+', color='black')

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
            fontproperties=fm.FontProperties(size=12),
        )
        ax.add_artist(scalebar)

        # Add a color bar
        cbar = add_colorbar_to_ax(img, fig, ax, where='bottom')
        cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
        cbar.ax.set_xlabel(
            r'Correlation with target pixel (in green)', fontsize=12
        )
        cbar.ax.xaxis.set_label_position('bottom')

        # Save plot as a PDF
        fig.tight_layout()
        file_path = plots_dir / f'{dataset}.pdf'
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.025)

        print(f'Done! ({time.time() - start_time:.1f} seconds)', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
