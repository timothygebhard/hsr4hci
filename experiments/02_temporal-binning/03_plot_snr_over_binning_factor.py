"""
Read in the results for different algorithms and different amounts of
temporal binning and plot the resulting SNR over the binning factor.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import time

import matplotlib.pyplot as plt
import pandas as pd

from hsr4hci.config import get_hsr4hci_dir
from hsr4hci.data import load_metadata
from hsr4hci.plotting import adjust_luminosity


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nPLOT SNR OVER BINNING FACTOR\n', flush=True)

    # -------------------------------------------------------------------------
    # Set up parser and get command line arguments
    # -------------------------------------------------------------------------

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--planet',
        type=str,
        default='b',
    )
    args = parser.parse_args()

    # Get arguments
    dataset = args.dataset
    planet = args.planet

    # -------------------------------------------------------------------------
    # Define shortcuts
    # -------------------------------------------------------------------------

    # Define directories from which we read (or to which we save) data
    experiments_dir = get_hsr4hci_dir() / 'experiments'
    dataset_dir = experiments_dir / '02_temporal-binning' / dataset
    pca_dir = dataset_dir / 'pca'
    signal_fitting_dir = dataset_dir / 'signal_fitting'
    signal_masking_dir = dataset_dir / 'signal_masking'

    # Load the metadata of the data set (e.g., because we need the DIT)
    print('Loading data set metadata...', end=' ', flush=True)
    metadata = load_metadata(name_or_path=dataset)
    dit = metadata['DIT_STACK']
    print('Done!', flush=True)

    # Initialize a new plot to which we will add everything
    fig, ax1 = plt.subplots(figsize=(7.24551, 7.24551 / 2))
    max_factor = 1

    # -------------------------------------------------------------------------
    # Load and plot the results for PCA
    # -------------------------------------------------------------------------

    # Read in the results for PCA into a pandas DataFrame
    print('Reading and plotting in PCA results...', end=' ', flush=True)

    try:

        # Read in the results for PCA into a pandas DataFrame
        file_path = pca_dir / f'metrics__{planet}.tsv'
        df = pd.read_csv(file_path, sep='\t')

        # Plot results for different numbers of principal components
        for i, n_components in enumerate([1, 5, 10, 20, 50, 100]):

            # Select the subset of the data frame that corresponds
            df_selection = df[df['n_components'] == n_components]
            max_factor = max(max_factor, df_selection.factor.max())

            # Plot the SNR over the binning factor
            ax1.plot(
                df_selection.factor,
                df_selection.log_fpf_mean,
                color=f'C{i + 1}',
            )
            ax1.plot(
                df_selection.factor,
                df_selection.log_fpf_mean,
                'o',
                markerfacecolor=f'C{i + 1}',
                markeredgecolor='white',
                markersize=4,
                label=f'PCA (n={n_components})',
            )
            # ax1.fill_between(
            #     df_selection.factor,
            #     df_selection.log_fpf_min,
            #     df_selection.log_fpf_max,
            #     facecolor=f'C{i + 1}',
            #     alpha=0.25,
            # )

        print('Done!', flush=True)

    except FileNotFoundError:
        print('Failed!', flush=True)

    # -------------------------------------------------------------------------
    # Load and plot the results for HSR (signal fitting)
    # -------------------------------------------------------------------------

    print('Reading in HSR results (signal fitting)...', end=' ', flush=True)

    try:

        # Read in the results for signal fitting into a pandas DataFrame
        file_path = signal_fitting_dir / f'metrics__{planet}.tsv'
        df = pd.read_csv(file_path, sep='\t')
        max_factor = max(max_factor, df.factor.max())

        # Plot the SNR over the binning factor
        ax1.plot(df.factor, df.log_fpf_mean, ls='-', color='C0')
        ax1.plot(
            df.factor,
            df.log_fpf_mean,
            's',
            markerfacecolor='C0',
            markeredgecolor='white',
            markersize=4,
            label='HSR (signal fitting)',
        )
        # ax1.fill_between(
        #     df.factor,
        #     df.log_fpf_min,
        #     df.log_fpf_max,
        #     facecolor='C0',
        #     alpha=0.25,
        # )

        print('Done!', flush=True)

    except FileNotFoundError:
        print('Failed!', flush=True)

    # -------------------------------------------------------------------------
    # Load and plot the results for HSR (signal masking)
    # -------------------------------------------------------------------------

    print('Reading in HSR results (signal masking)...', end=' ', flush=True)

    try:

        # Read in the results for signal masking into a pandas DataFrame
        file_path = signal_masking_dir / f'metrics__{planet}.tsv'
        df = pd.read_csv(file_path, sep='\t')
        max_factor = max(max_factor, df.factor.max())

        # Plot the SNR over the binning factor
        ax1.plot(
            df.factor, df.log_fpf_mean, ls='--', color=adjust_luminosity('C0')
        )
        ax1.plot(
            df.factor,
            df.log_fpf_mean,
            's',
            markerfacecolor=adjust_luminosity('C0'),
            markeredgecolor='white',
            markersize=4,
            label='HSR (signal masking)',
        )
        # ax1.fill_between(
        #     df.factor,
        #     df.log_fpf_min,
        #     df.log_fpf_max,
        #     facecolor=adjust_luminosity('C0'),
        #     alpha=0.25,
        # )

        print('Done!', flush=True)

    except FileNotFoundError:
        print('Failed!', flush=True)

    # -------------------------------------------------------------------------
    # Set up plot options and save results
    # -------------------------------------------------------------------------

    ax1.set_xscale('log')

    # Add a secondary x-axis on top which converts the stacking factor to
    # an effective integration time by multiplying the stacking factor with
    # the DIT of a single frame (0.2 seconds for most L' band data sets)
    ax2 = ax1.secondary_xaxis(
        location='top', functions=(lambda x: dit * x, lambda x: x / dit)
    )

    ax1.grid(which='both', color='lightgray', ls='--')
    ax1.legend(
        loc='lower center',
        ncol=8,
        fontsize=6,
        handletextpad=0.05,
        mode="expand",
        columnspacing=1.5,
    )

    # Set plot limits
    ax1.set_xlim(0.9, 1.1 * max_factor)
    ax1.set_ylim(0.0, None)

    # Set up font sizes
    for ax in (ax1, ax2):
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(6)

    # Add labels
    ax1.set_xlabel('Binning factor')
    ax2.set_xlabel('Effective Integration Time (s)')
    ax1.set_ylabel('-log(False Positive Fraction)')
    fig.tight_layout()

    # Save plot
    file_path = dataset_dir / 'log_fpf_over_binning_factor.pdf'
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0.025)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
