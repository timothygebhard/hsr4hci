"""
Create a plot of the interpolated FPF values (which implicitly defines
the contrast curve).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Optional, Any

import argparse
import time

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import norm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hsr4hci.plotting import set_fontsize


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nPLOT FPF INTERPOLATION AND DETECTION LIMITS\n')

    # -------------------------------------------------------------------------
    # Parse command line arguments
    # -------------------------------------------------------------------------

    # Set up a parser and parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
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
    parser.add_argument(
        '--sigma-threshold',
        type=float,
        default=5.0,
        help='(Gaussian) sigma for contrast curve. Default: 5.',
    )
    args = parser.parse_args()

    # Define shortcuts
    dataset = args.dataset
    mode = args.mode
    sigma_threshold = args.sigma_threshold

    # Get path to main directory
    main_dir = Path(dataset).resolve()
    if not main_dir.exists():
        raise RuntimeError(f'{main_dir} does not exist!')

    # -------------------------------------------------------------------------
    # Define threshold; prepare new plot
    # -------------------------------------------------------------------------

    def transform(x: Any) -> Any:
        return -np.log10(x)

    # Define significance level and convert to logarithmic space
    threshold = float(transform(1 - norm.cdf(sigma_threshold, 0, 1)))
    threshold_label = rf'{sigma_threshold}$\sigma$ threshold'

    # Create a new figure
    fig, axes = plt.subplots(
        nrows=7,
        figsize=(9 / 2.54, 20.15 / 2.54),
        sharex='all',
    )
    fig.subplots_adjust(
        left=0.12, right=0.995, top=0.95, bottom=0.04, hspace=0.1
    )
    axes[0].set_xticks(np.arange(5, 13))
    axes[-1].set_xlabel('Contrast of injected planet (in magnitudes) ')

    # Setup various options for all axes
    for ax in axes:
        set_fontsize(ax=ax, fontsize=6)
        ax.axhline(y=threshold, ls='--', color='black', lw=1)
        ax.set_ylim(0, 100)
        ax.set_ylabel(r'$-\mathrm{log}_\mathrm{10}(\mathrm{FPF})$')
        ax.set_xlim(4.5, 12.5)
        for k, spine in ax.spines.items():
            spine.set_zorder(1000)

    # Disable bottom ticks for all plots but the last one
    for ax in axes[:-1]:
        ax.tick_params(bottom=False)

    # -------------------------------------------------------------------------
    # Loop over different result files and create plots
    # -------------------------------------------------------------------------

    # Store handles for legend
    handles = [
        Line2D([0], [0], alpha=0, lw=1, label='    Legend:'),
        Line2D([0], [0], alpha=0, lw=1, label='    '),
        Line2D([0], [0], color='k', lw=1, ls='--', label=threshold_label),
    ]

    # Loop over different algorithms
    for name, path, color in [
        ('PCA (n=20)', main_dir / 'pca-20', 'C0'),
        ('HSR (signal fitting)', main_dir / 'signal_fitting', 'C1'),
        ('HSR (signal masking)', main_dir / 'signal_masking', 'C2'),
    ]:

        # Read in result
        file_path = Path(path) / f'results__{mode}.tsv'
        if not file_path.exists():
            continue
        else:
            df = pd.read_csv(file_path, sep='\t')

        # Get the available expected contrasts
        expected_contrasts = np.array(sorted(np.unique(df.expected_contrast)))

        # Loop over different separations
        for i, separation in enumerate(np.arange(2, 9)):

            # Select the axis for this separation
            ax = axes[i]

            label = ax.text(
                x=0.965,
                y=0.90,
                s=f'Separation: {separation} FWHM',
                ha='right',
                va='top',
                transform=ax.transAxes,
                fontsize=6,
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    fc='white',
                    ec='black',
                    lw=0.5,
                ),
                zorder=100,
            )

            # Store y values for the interpolation (i.e., the mean / median
            # of the FPF / logFPF)
            average_values = []

            # Loop over the expected contrast
            for expected_contrast in expected_contrasts:

                # Select the subset of the data frame that matches the current
                # separation and expected contrast. This should contain one
                # entry for each azimuthal position (i.e., usually 6).
                df_subset = df[
                    (df.separation == separation)
                    & (df.expected_contrast == expected_contrast)
                ]['fpf_mean']

                # Compute the average value and add it to the plot; also plot
                # the six data points to give an idea of the variance
                average_value = np.median(transform(df_subset))
                ax.plot(
                    len(df_subset) * [expected_contrast],
                    transform(df_subset),
                    ls='none',
                    marker='.',
                    color=color,
                    ms=1,
                )

                # Store the average logFPF value for the interpolator
                average_values.append(average_value)

            # Set up a linear (k=1) spline interpolator so that we can
            # estimate the (log)FPF value at arbitrary contrast value
            interpolator = InterpolatedUnivariateSpline(
                x=expected_contrasts, y=np.array(average_values), k=1
            )

            # Define a (very fine) grid of contrast values, evaluate the
            # interpolator on this grid, and plot the result
            grid = np.linspace(
                min(expected_contrasts), max(expected_contrasts), int(1e4)
            )
            ax.plot(grid, interpolator(grid), color=color, lw=1)
            ax.plot(
                expected_contrasts, np.array(average_values), '.', color=color
            )

            # Define a helper function to find the (maximum) index after
            # which the values of `array` change their sign
            # Source: https://stackoverflow.com/a/21468492/4100721
            def get_root_idx(array: np.ndarray) -> Optional[int]:
                a, b = array > 0, array <= 0
                idx = ((a[:-1] & b[1:]) | (b[:-1] & a[1:])).nonzero()[0]
                return int(np.max(idx)) if idx.size > 0 else None

            # Get the index of the grid entry where the interpolated FPF
            # or logFPF values cross the given `threshold`
            idx = get_root_idx(np.array(interpolator(grid)) - threshold)
            if idx is not None:

                # If it exists, use this index to compute the *contrast* value
                # at which the FPF / logFPF crosses the threshold, and plot it
                threshold_contrast = 0.5 * (grid[idx] + grid[idx + 1])
                ax.axvline(
                    x=threshold_contrast, color=color, lw=1.5, zorder=99
                )

        # Add handles for the legend
        handles += [Patch(fc=color, ec='none', label=name)]

    # -------------------------------------------------------------------------
    # Add legend to the plot and save the plot
    # -------------------------------------------------------------------------

    # Add legend to the plot
    fig.legend(
        handles=handles,
        ncol=3,
        mode='expand',
        loc=9,
        fontsize=6,
        frameon=False,
    )

    # Save plot as PDF
    plt.savefig(main_dir / f'fpf-interpolation__{mode}.pdf', dpi=600)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
