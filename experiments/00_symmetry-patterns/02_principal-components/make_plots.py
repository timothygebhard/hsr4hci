"""
Run PCA and plot the principal components / eigenimages.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import time

import matplotlib.pyplot as plt
import numpy as np

from hsr4hci.data import load_dataset
from hsr4hci.plotting import plot_frame
from hsr4hci.pca import get_pca_signal_estimates


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nRUN PCA AND PLOT PRINCIPAL COMPONENTS\n', flush=True)

    # -------------------------------------------------------------------------
    # Run PCA on all data sets and plot the principal components / eigenimages
    # -------------------------------------------------------------------------

    # Ensure the plots directory exists
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    # Loop over different data sets
    for dataset in (
        'beta_pictoris__lp',
        'beta_pictoris__mp',
        'hr_8799__lp',
        'r_cra__lp',
    ):

        start_time = time.time()
        print(f'Running for {dataset}...', end=' ', flush=True)

        # Load data set (and crop to some reasonable size)
        stack, parang, psf_template, obs_con, metadata = load_dataset(
            name_or_path=dataset,
            frame_size=(51, 51),
            binning_factor=1,
        )
        n_frames, x_size, y_size = stack.shape

        # Run PCA and compute principal components
        _, components = get_pca_signal_estimates(
            stack=stack,
            parang=parang,
            n_components=24,
            return_components=True,
        )

        # Loop over principal components and plot them
        for n in range(24):

            # Scale components to (-1, 1)
            plot_array = components[n]
            plot_array /= np.nanpercentile(np.abs(plot_array), 99.99)

            # Create plot
            fig, ax, cbar = plot_frame(
                frame=plot_array,
                positions=[],
                labels=[],
                pixscale=float(metadata['PIXSCALE']),
                figsize=(2.125 / 2.54, 2.125 / 2.54),
                subplots_adjust=dict(
                    left=0.01, right=0.99, bottom=0.01, top=0.99,
                ),
                aperture_radius=0,
                scalebar_color='black',
                limits=(-1, 1),
                add_colorbar=False,
            )

            # Add number of the principal component
            ax.annotate(
                f'PC #{n}',
                xy=(51 - 5, 4),
                xycoords='data',
                xytext=(51 - 5, 4),
                textcoords='data',
                ha='right',
                va='bottom',
                fontsize=6,
            )

            # Ensure that the results directory for this data set exists
            dataset_dir = plots_dir / dataset
            dataset_dir.mkdir(exist_ok=True)

            # Save the plot as a PDF
            file_path = dataset_dir / f'{dataset}__n_pc={n}.pdf'
            plt.savefig(file_path, dpi=600)
            plt.close()

        print(f'Done! ({time.time() - start_time:.1f} seconds)', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
