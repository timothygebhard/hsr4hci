"""
Collect results for experiments of an algorithm.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from itertools import product
from pathlib import Path

import argparse
import time

from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from hsr4hci.contrast import get_contrast_curve
from hsr4hci.plotting import set_fontsize


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nPLOT BRIGHTNESS RATIO TABLE AND CONTRAST CURVES\n')

    # -------------------------------------------------------------------------
    # Parse command line arguments
    # -------------------------------------------------------------------------

    # Set up a parser and parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        required=True,
        help='Main directory of the experiment set.',
    )
    parser.add_argument(
        '--mode',
        choices=['classic', 'alternative'],
        default='classic',
        help=(
            'How to compute the throughput: using the "classic" approach'
            'where the no_fake_planets residual is subtracted, or the '
            'alternative approach that estimate the background from other '
            'positions at the same separation.'
        ),
    )
    args = parser.parse_args()
    mode = args.mode
    main_dir = Path(args.directory).resolve()
    if not main_dir.exists():
        raise RuntimeError(f'{main_dir} does not exist!')

    # -------------------------------------------------------------------------
    # Load the results.tsv file; create pivot tables and labels
    # -------------------------------------------------------------------------

    file_path = main_dir / f'results__{mode}.tsv'
    results_df = pd.read_csv(file_path, sep='\t')

    pivot_median = results_df.pivot_table(
        index='expected_contrast',
        columns='separation',
        values='throughput',
        aggfunc={'throughput': np.nanmedian},
    )
    pivot_min = results_df.pivot_table(
        index='expected_contrast',
        columns='separation',
        values='throughput',
        aggfunc={'throughput': np.min},
    )
    pivot_max = results_df.pivot_table(
        index='expected_contrast',
        columns='separation',
        values='throughput',
        aggfunc={'throughput': np.max},
    )

    labels = np.full(shape=pivot_median.shape, fill_value='').tolist()

    n_row, n_col = pivot_median.shape
    for i, j in product(range(n_row), range(n_col)):
        median = pivot_median.values[i, j]
        lower = pivot_median.values[i, j] - pivot_min.values[i, j]
        upper = pivot_max.values[i, j] - pivot_median.values[i, j]
        labels[i][j] = rf'${median:.2f}_{{-{lower:.2f}}}^{{+{upper:.2f}}}$'

    # -------------------------------------------------------------------------
    # Create the heatmap-like plot of the brightness ratio table
    # -------------------------------------------------------------------------

    # Create a new plot
    fig, ax = plt.subplots(figsize=(9 / 2.54, 6 / 2.54))
    fig.subplots_adjust(left=0.12, right=1, top=1, bottom=0.135)

    # Create a heatmap plot of the pivot table
    heatmap = sns.heatmap(
        ax=ax,
        data=pivot_median.values,
        annot=labels,
        fmt='',
        annot_kws={'size': 5, 'alpha': 0.65},
        xticklabels=1,
        yticklabels=1,
        vmin=0,
        vmax=1,
        cmap='GnBu',
        linewidths=0.25,
        cbar=False,
    )

    # Add the right ticks / tick labels
    ax.set_xticklabels(np.unique(results_df.separation.values))
    ax.set_yticklabels(np.unique(results_df.expected_contrast.values))

    # Add a hatched patch for all throughput values greater 1 which need to
    # be interpreted with extra caution / discussed separately
    for i, (expected_contrast, row) in enumerate(pivot_median.iterrows()):
        for j, (separation, throughput) in enumerate(row.items()):
            if throughput > 1.15:
                ax.add_patch(
                    Rectangle(
                        (j, i),
                        1,
                        1,
                        ec='white',
                        fc='none',
                        lw=0.1,
                        hatch='////////',
                        alpha=0.35,
                    )
                )

    # -------------------------------------------------------------------------
    # Compute and plot the contrast curve
    # -------------------------------------------------------------------------

    # Define the sigma threshold which we will use to compute the limits
    sigma_threshold = 5

    # Compute the contrast curve
    separations, detection_limits = get_contrast_curve(
        df=results_df,
        sigma_threshold=sigma_threshold,
        log_transform=True,
    )

    # Plot the contrast curve
    ax.plot(
        separations + 0.5 - min(separations),
        detection_limits,
        linewidth=3,
        color='white',
        solid_capstyle='round',
        zorder=99,
    )
    ax.plot(
        separations + 0.5 - min(separations),
        detection_limits,
        '.-',
        markeredgecolor='white',
        markeredgewidth=0.5,
        linewidth=2,
        color='orangered',
        solid_capstyle='round',
        zorder=100,
    )

    # -------------------------------------------------------------------------
    # Add plot options
    # -------------------------------------------------------------------------

    # Set up font sizes for the whole plot
    set_fontsize(ax=ax, fontsize=6)

    # Set up plot labels
    ax.set_xlabel('Separation (in units of FWHM)')
    ax.set_ylabel('Contrast (in magnitudes)')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Save the final plot
    file_path = main_dir / f'{main_dir.name}__{mode}.pdf'
    plt.savefig(file_path, dpi=600)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
