"""
Read in the results for different algorithms and different amounts of
temporal binning and plot the -log(FPF) over the binning factor.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hsr4hci.data import load_metadata
from hsr4hci.plotting import adjust_luminosity, set_fontsize


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nPLOT -LOG(FPF) OVER BINNING FACTOR\n', flush=True)

    # -------------------------------------------------------------------------
    # Set up parser and get command line arguments
    # -------------------------------------------------------------------------

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset, e.g., "beta_pictoris__lp".',
    )
    parser.add_argument(
        '--planet',
        type=str,
        default='b',
        help='Planet, e.g., "b".',
    )
    args = parser.parse_args()

    # Get arguments
    dataset = args.dataset
    planet = args.planet

    # -------------------------------------------------------------------------
    # Define shortcuts
    # -------------------------------------------------------------------------

    # Define directory for the dataset that we are processing
    dataset_dir = Path(__file__).resolve().parent / dataset

    # Load the metadata of the data set (e.g., because we need the DIT)
    print('Loading data set metadata...', end=' ', flush=True)
    metadata = load_metadata(name_or_path=dataset)
    dit = metadata['DIT_STACK']
    print('Done!', flush=True)

    # Initialize a new plot to which we will add everything
    fig, ax1 = plt.subplots(figsize=(18.4 / 2.54, 18.4 / 2.54 / 2.5))
    fig.subplots_adjust(left=0.052, right=0.998, top=0.88, bottom=0.12)

    # -------------------------------------------------------------------------
    # Load and plot the results for PCA
    # -------------------------------------------------------------------------

    print('Processing PCA results...', end=' ', flush=True)

    # Read in the results for PCA into a pandas DataFrame
    file_path = dataset_dir / 'pca' / f'metrics__{planet}.tsv'
    df = pd.read_csv(file_path, sep='\t')

    # Get the maximum binning factor (for plot limits)
    max_binning_factor = df.binning_factor.max()

    # Plot results for different numbers of principal components
    for i, n_components in enumerate([1, 5, 10, 20, 50, 100]):

        # Select the subset of the data frame for the current n_components
        df_selection = df[df['n_components'] == n_components]

        # Select results and sort by binning factor
        idx = np.argsort(df_selection.binning_factor.values)
        binning_factor = df_selection.binning_factor.values[idx]
        log_fpf_mean = df_selection.log_fpf_mean.values[idx]

        # Plot the -log(FPF) over the binning factor
        ax1.plot(binning_factor, log_fpf_mean, color=f'C{i + 1}')
        ax1.plot(
            binning_factor,
            log_fpf_mean,
            'o',
            markerfacecolor=f'C{i + 1}',
            markeredgecolor='white',
            markersize=4,
            label=f'PCA (n={n_components})',
        )

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Load and plot the results for HSR (signal fitting)
    # -------------------------------------------------------------------------

    print('Processing HSR results (signal fitting)...', end=' ', flush=True)

    # Read in the results for signal fitting into a pandas DataFrame
    file_path = dataset_dir / 'signal_fitting' / f'metrics__{planet}.tsv'
    df = pd.read_csv(file_path, sep='\t')

    # Select results and sort by binning factor
    idx = np.argsort(df.binning_factor.values)
    binning_factor = df.binning_factor.values[idx]
    log_fpf_mean = df.log_fpf_mean.values[idx]

    # Plot the -log(FPF) over the binning factor
    ax1.plot(binning_factor, log_fpf_mean, ls='-', color='C0')
    ax1.plot(
        binning_factor,
        log_fpf_mean,
        's',
        markerfacecolor='C0',
        markeredgecolor='white',
        markersize=4,
        label='HSR (signal fitting)',
    )

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Load and plot the results for HSR (signal masking)
    # -------------------------------------------------------------------------

    print('Processing HSR results (signal masking)...', end=' ', flush=True)

    # Read in the results for signal masking into a pandas DataFrame
    file_path = dataset_dir / 'signal_masking' / f'metrics__{planet}.tsv'
    df = pd.read_csv(file_path, sep='\t')

    # Select results and sort by binning factor
    idx = np.argsort(df.binning_factor.values)
    binning_factor = df.binning_factor.values[idx]
    log_fpf_mean = df.log_fpf_mean.values[idx]

    # Plot the -log(FPF) over the binning factor
    ax1.plot(
        binning_factor,
        log_fpf_mean,
        ls='--',
        color=adjust_luminosity('C0'),
    )
    ax1.plot(
        binning_factor,
        log_fpf_mean,
        's',
        markerfacecolor=adjust_luminosity('C0'),
        markeredgecolor='white',
        markersize=4,
        label='HSR (signal masking)',
    )

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Set up plot options and save results
    # -------------------------------------------------------------------------

    # Add a secondary x-axis on top which converts the binning factor to
    # an effective integration time by multiplying the binning factor with
    # the DIT of a single frame (= 0.2 seconds for most L' band data sets)
    ax2 = ax1.secondary_xaxis(
        location='top', functions=(lambda x: dit * x, lambda x: x / dit)
    )

    # Add a grid and a legend to the plot
    ax1.grid(which='both', color='lightgray', ls='--')
    ax1.legend(
        loc='lower center',
        ncol=8,
        fontsize=6,
        handletextpad=0.05,
        mode="expand",
        columnspacing=1.5,
    )

    # Set axes scale and limits
    ax1.set_xscale('log')
    ax1.set_xlim(0.9, 1.1 * max_binning_factor)
    ax1.set_ylim(0.0, None)

    # Set up font sizes
    for ax in (ax1, ax2):
        set_fontsize(ax=ax, fontsize=6)

    # Add labels to axes
    ax1.set_xlabel('Binning factor')
    ax2.set_xlabel('Effective Integration Time (s)')
    ax1.set_ylabel(r'$-\log_\mathrm{10}(\mathrm{FPF})$')

    # Save plot
    file_path = dataset_dir / 'log_fpf_over_binning_factor.pdf'
    plt.savefig(file_path, dpi=600)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
