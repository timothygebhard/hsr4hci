"""
This script plots the SNR as a function of the number of principal
components, using the figures_of_merit.csv results file created by
04_compute_snrs.py.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Dict, List, Tuple

import csv
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hsr4hci.utils.argparsing import get_base_directory
from hsr4hci.utils.plotting import adjust_luminosity


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nPLOT SNR OVER NUMBER OF PRINCIPAL COMPONENTS\n', flush=True)

    # -------------------------------------------------------------------------
    # Parse command line arguments and load config.json
    # -------------------------------------------------------------------------

    # Get base_directory from command line arguments
    base_dir = get_base_directory()

    # Read in the config file and parse it
    file_path = base_dir / 'config.json'
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)

    # -------------------------------------------------------------------------
    # Define some shortcuts
    # -------------------------------------------------------------------------

    # Define shortcuts for planet keys
    planet_keys = list(config['evaluation']['planets'].keys())

    # Shortcuts to other entries in the configuration
    dit = float(config['metadata']['DIT'])
    plot_size = config['evaluation']['plot_size']
    min_n_components = config['pca']['min_n_components']
    max_n_components = config['pca']['max_n_components']

    # Construct numbers of principal components (and count them)
    pc_numbers = list(range(min_n_components, max_n_components + 1))

    # Other shortcuts
    baselines_dir = base_dir / 'pca_baselines'

    # -------------------------------------------------------------------------
    # Make plots for different stacking factors
    # -------------------------------------------------------------------------

    # Store maximum SNR for each stacking factor (and each planet)
    max_snrs: Dict[str, List[Tuple[int, float, float]]] = {
        key: [] for key in planet_keys
    }

    # Run for each stacking factor
    for stacking_factor in config['stacking_factors']:

        print(
            f'Running for stacking factor {stacking_factor}...',
            end=' ',
            flush=True,
        )

        # Construct path to result dir for this stacking factor
        result_dir = baselines_dir / f'stacked_{stacking_factor}'

        # Read in the CSV file with the figures of merit into a dataframe
        file_path = result_dir / 'figures_of_merit.csv'
        dataframe = pd.read_csv(
            filepath_or_buffer=file_path,
            sep='\t',
            header=[0, 1],
            index_col=[0, 1],
        )

        # Set up a figure
        plt.figure(figsize=tuple(plot_size))

        # Make plot for each planet individually
        for i, planet_key in enumerate(planet_keys):

            # Construct full name of the planet (target star + letter)
            planet_name = f'{config["metadata"]["TARGET_STAR"]} {planet_key}'

            # Select the SNR values from the dataframe
            snr_values = dataframe[planet_key]['snr'].values

            # Get maximum SNR and draw it separately
            max_snr = float(np.max(snr_values))
            max_snr_idx = int(np.argmax(snr_values))
            plt.plot(
                pc_numbers[max_snr_idx],
                max_snr,
                color=adjust_luminosity(f'C{i}'),
                marker='x',
                ms=10,
            )

            # Store the maximum SNR
            max_snrs[planet_key].append(
                (stacking_factor, stacking_factor * dit, max_snr)
            )

            # Plot the SNR as a step function
            plt.step(
                pc_numbers,
                snr_values,
                'o-',
                color=f'C{i}',
                where='mid',
                label=planet_name,
            )

            # Add a label for each data point
            for (n, snr) in zip(pc_numbers, snr_values):
                plt.annotate(
                    text=f'{snr:.2f}\n{n:d}',
                    xy=(n, snr),
                    ha='center',
                    va='center',
                    fontsize=1.5,
                )

        # Determine maximum SNR for plot limits
        snr_idx = dataframe.columns.get_level_values(1) == 'snr'
        max_snr = np.max(np.nan_to_num(dataframe.iloc[:, snr_idx].values))

        # Add plot options
        plt.xlim(min_n_components - 1, max_n_components + 1)
        plt.ylim(0, 1.1 * max_snr)
        plt.xlabel('Number of PCA components')
        plt.ylabel('Signal-to-noise ratio (SNR)')
        plt.title(f'PynPoint baseline for stacked_{stacking_factor}')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.xticks(list(range(min_n_components - 1, max_n_components + 1, 10)))
        plt.grid(which='both', ls='--', color='LightGray', alpha=0.5)

        # Save plot as a PDF
        file_path = result_dir / 'snr_over_npc.pdf'
        plt.savefig(file_path, bbox_inches='tight')
        plt.clf()
        plt.close()

        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Store maximum SNR values for each planet as a CSV file
    # -------------------------------------------------------------------------

    # Create folder to store results
    results_dir = baselines_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    print('\nSaving SNRs to CSV file...', end=' ', flush=True)
    for planet_key in planet_keys:
        file_path = results_dir / f'snrs__{planet_key}.csv'
        with open(file_path, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Stacking factor', 'Effective DIT', 'SNR'])
            csv_writer.writerows(max_snrs[planet_key])
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create plot of maximum SNR as a function of the stacking factor
    # -------------------------------------------------------------------------

    print('Plotting SNR over stacking factor...', end=' ', flush=True)

    # Create a new figure
    fig, ax1 = plt.subplots()

    for i, planet_key in enumerate(planet_keys):

        # Define shortcuts
        color = f'C{i}'
        planet_name = f'{config["metadata"]["TARGET_STAR"]} {planet_key}'

        # Get stacking factors (x) and SNRs (y)
        x, _, y = zip(*max_snrs[planet_key])

        # Plot the SNR over the stacking factor
        ax1.plot(x, y, color=color, alpha=0.3)
        ax1.plot(x, y, 'x', color=color, label=planet_name)

    # Add a secondary x-axis on top which converts the stacking factor to
    # an effective integration time by multiplying the stacking factor with
    # the DIT of a single frame (0.2 seconds for most L' band data sets)
    ax2 = ax1.secondary_xaxis(
        location='top', functions=(lambda x: dit * x, lambda x: x / dit)
    )

    # Add axes labels
    ax1.set_xlabel('Stacking factor')
    ax1.set_ylabel('Signal-to-noise ratio (SNR)')
    ax2.set_xlabel('Effective Integration Time (s)')

    # Set limits and other plotting options
    ax1.set_xlim(0, None)
    ax1.set_ylim(0, None)
    plt.grid(color='lightgray', ls='--')
    plt.legend(loc='best')
    plt.tight_layout()

    # Save the plot
    plt.savefig(results_dir / 'snr_over_stacking_factor.pdf', dpi=300)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
