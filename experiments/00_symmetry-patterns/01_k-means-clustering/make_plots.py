"""
Apply clustering to time series and plot the results.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import time

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from hsr4hci.coordinates import get_center
from hsr4hci.data import load_dataset
from hsr4hci.plotting import disable_ticks


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

    # Ensure the plots directory exists
    plots_dir = Path('plots')
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
        center = get_center((stack.shape[1], stack.shape[2]))

        # Normalize / whiten the data
        # This is necessary, otherwise everything is just clustered based on
        # the scale of the values, meaning we get rings around the star...
        sample = StandardScaler().fit_transform(stack.reshape(n_frames, -1))

        # Run k-means clustering on time series for 10 clusters
        clustering = KMeans(n_clusters=10).fit_predict(sample.T)

        # Reshape the clustering result to the original frame size
        plot_array = clustering.reshape(x_size, y_size)

        # Prepare grid for the pcolormesh()
        x_range = np.arange(x_size)
        y_range = np.arange(y_size)
        x, y = np.meshgrid(x_range, y_range)

        # Plot the result
        fig, ax = plt.subplots(figsize=(5, 5))
        img = ax.pcolormesh(
            x,
            y,
            plot_array,
            shading='nearest',
            cmap='tab20',
            rasterized=True,
        )
        ax.plot(center[0], center[1], '+', color='black', ms=12)
        disable_ticks(ax)

        # Create the scale bar and add it to the frame
        scalebar = AnchoredSizeBar(
            transform=ax.transData,
            size=0.5 / float(metadata['PIXSCALE']),
            label='0.5"',
            loc=2,
            pad=1,
            color='white',
            frameon=False,
            size_vertical=0,
            fontproperties=fm.FontProperties(size=12),
        )
        ax.add_artist(scalebar)

        # Save the plot as a PDF
        fig.tight_layout()
        file_path = plots_dir / f'{dataset}.pdf'
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.025)

        print(f'Done! ({time.time() - start_time:.1f} seconds)', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
