"""
Apply clustering to time series and plot the results.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import time

from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np

from hsr4hci.config import get_experiments_dir
from hsr4hci.data import load_dataset
from hsr4hci.plotting import plot_frame


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nRUN K-MEANS CLUSTERING EXPERIMENT AND MAKE PLOTS\n', flush=True)

    # -------------------------------------------------------------------------
    # Run k-means clustering for two example data sets
    # -------------------------------------------------------------------------

    # Fix seed so that clustering becomes deterministic
    np.random.seed(42)

    # Ensure the plots directory exists
    plots_dir = (
        get_experiments_dir()
        / 'appendix'
        / 'A.3_k-means-clustering'
        / 'plots'
    )
    plots_dir.mkdir(exist_ok=True)

    # Loop over different data sets
    for dataset in ('beta_pictoris__lp', 'r_cra__lp'):

        start_time = time.time()
        print(f'Running for {dataset}...', end=' ', flush=True)

        # Load data set (and crop to some reasonable size)
        stack, parang, psf_template, obs_con, metadata = load_dataset(
            name_or_path=dataset,
            frame_size=(51, 51),
        )
        n_frames, x_size, y_size = stack.shape

        # Normalize / whiten the data
        # This is necessary, otherwise everything is just clustered based on
        # the scale of the values, meaning we get rings around the star...
        sample = StandardScaler().fit_transform(stack.reshape(n_frames, -1))

        # Run k-means clustering on time series for 10 clusters
        clustering = KMeans(n_clusters=10).fit_predict(sample.T)

        # Reshape the clustering result to the original frame size
        plot_array = clustering.reshape(x_size, y_size)

        # Create plot
        fig, ax, cbar = plot_frame(
            frame=plot_array,
            positions=[],
            labels=[],
            pixscale=float(metadata['PIXSCALE']),
            figsize=(4.3 / 2.54, 5.3 / 2.54),
            subplots_adjust=dict(
                left=0.001,
                right=0.999,
                top=0.855,
                bottom=0.005,
            ),
            aperture_radius=0,
            scalebar_color='black',
            scalebar_loc='upper left',
            add_colorbar=False,
            cmap='tab20',
            limits=(0, 10),
            add_cardinal_directions=False,
        )

        # Add empty colorbar on top of the axis to ensure spacing, margins,
        # etc. are identical to the correlation map plots
        img = ax.collections[0]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('top', size='5%', pad=0.025)
        cax.set_axis_off()

        # Save the plot as a PDF
        file_path = plots_dir / f'{dataset}.pdf'
        plt.savefig(file_path, dpi=600)

        print(f'Done! ({time.time() - start_time:.1f} seconds)', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
