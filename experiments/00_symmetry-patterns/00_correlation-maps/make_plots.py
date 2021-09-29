"""
Create correlation map plots.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import time

from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt

from hsr4hci.data import load_metadata
from hsr4hci.fits import read_fits
from hsr4hci.plotting import plot_frame


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

        # Load metadata for dataset
        metadata = load_metadata(name_or_path=dataset)

        # Load correlation coefficients from FITS
        file_path = fits_dir / f'{dataset}.fits'
        correlations = read_fits(file_path=file_path, return_header=False)

        # Select CC map that we want to plot (flip coordinates because numpy!)
        cc_map = correlations[y, x]

        # Create plot
        fig, ax, _ = plot_frame(
            frame=cc_map,
            positions=[],
            labels=[],
            pixscale=metadata['PIXSCALE'],
            figsize=(4.3 / 2.54, 5.3 / 2.54),
            subplots_adjust=dict(
                left=0.001,
                right=0.999,
                top=0.855,
                bottom=0.005,
            ),
            aperture_radius=0,
            scalebar_color='black',
            scalebar_loc=2,
            limits=(-0.6, 0.6),
            add_colorbar=False,
        )

        # Add target pixel
        ax.plot(x, y, 'x', ms=4, color='white')

        # Add colorbar on top of the axis
        img = ax.collections[0]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('top', size='5%', pad=0.025)
        cbar = fig.colorbar(
            img, cax=cax, orientation='horizontal', ticklocation='top'
        )
        cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
        cbar.ax.set_xlabel('Correlation with target pixel', fontsize=6)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(labelsize=5)

        # Save plot as a PDF
        file_path = plots_dir / f'{dataset}.pdf'
        plt.savefig(file_path, dpi=600)

        print(f'Done! ({time.time() - start_time:.1f} seconds)', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
