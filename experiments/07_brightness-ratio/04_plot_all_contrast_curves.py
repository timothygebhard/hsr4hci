"""
Create a plot of the interpolated FPF values, which implicitly defines
the contrast curve.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import time

import matplotlib.pyplot as plt
import pandas as pd

from hsr4hci.contrast import get_contrast_curve
from hsr4hci.plotting import set_fontsize, adjust_luminosity


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nPLOT ALL CONTRAST CURVES\n')

    # -------------------------------------------------------------------------
    # Parse command line arguments
    # -------------------------------------------------------------------------

    # Set up a parser and parse the command line arguments
    parser = argparse.ArgumentParser()
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

    # Define shortcuts
    mode = args.mode

    # -------------------------------------------------------------------------
    # Prepare plot
    # -------------------------------------------------------------------------

    # Create a new figure and adjust margins
    fig, ax = plt.subplots(figsize=(9 / 2.54, 4.5 / 2.54))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.96, bottom=0.18)

    # Add axes labels and set limits
    ax.set_xlabel('Separation (in units of the PSF FWHM)')
    ax.set_ylabel('Contrast (in magnitudes)')
    ax.set_xlim(1.75, 8.25)
    ax.set_ylim(11, 6)
    ax.set_yticks([6, 7, 8, 9, 10, 11])

    # Adjust fontsize and add grid
    set_fontsize(ax=ax, fontsize=6)
    ax.grid(
        b=True,
        which='both',
        lw=1,
        alpha=0.3,
        dash_capstyle='round',
        dashes=(0, 2),
    )

    # -------------------------------------------------------------------------
    # Loop over different result files and plot contrast curves
    # -------------------------------------------------------------------------

    # Loop over different algorithms
    for name, path, color, ls in [
        ('PCA (n=20)', 'pca', 'black', '-'),
        ('HSR (signal fitting)', 'signal_fitting', 'C0', '-'),
        ('HSR (signal masking)', 'signal_masking', 'C1', '-'),
        ('HSR (signal fitting + OC)', 'signal_fitting__oc', 'C0', '--'),
        ('HSR (signal masking + OC)', 'signal_masking__oc', 'C1', '--'),
    ]:

        print(f'Plotting {name}...', end=' ', flush=True)

        # Read in result
        file_path = Path('./experiments/') / path / f'results__{mode}.tsv'
        df = pd.read_csv(file_path, sep='\t')

        # Compute the contrast curve
        separations, detection_limits = get_contrast_curve(
            df=df, sigma_threshold=5, log_transform=True
        )

        # Define additional arguments for plotting
        if ls == '--':
            plot_args = dict(dash_capstyle='round', dashes=(1, 2), color=color)
        else:
            plot_args = dict(color=adjust_luminosity(color))

        # Plot the contrast curve
        ax.plot(separations, detection_limits, ls=ls, label=name, **plot_args)

        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Add legend to the plot and save the plot
    # -------------------------------------------------------------------------

    # Add legend
    ax.legend(loc='upper right', fontsize=6)

    # Save plot as PDF
    plt.savefig(f'all-contrast-curves__{mode}.pdf', dpi=600)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
