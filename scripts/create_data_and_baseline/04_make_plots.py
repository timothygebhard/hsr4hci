"""
This script plots the SNR as a function of the number of principal
components, using the figures_of_merit.csv results file created by
03_compute_snrs.py.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import json
import os
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

    # Construct (expected) path to config.json
    file_path = os.path.join(base_dir, 'config.json')

    # Read in the config file and parse it
    with open(file_path, 'r') as config_file:
        config = json.load(config_file)

    # -------------------------------------------------------------------------
    # Define some shortcuts
    # -------------------------------------------------------------------------

    # Shortcuts to entries in the configuration
    planet_positions = config['evaluation']['planet_positions']
    plot_size = config['evaluation']['plot_size']
    min_n_components = config['pca']['min_n_components']
    max_n_components = config['pca']['max_n_components']

    # Construct numbers of principal components (and count them)
    pc_numbers = list(range(min_n_components, max_n_components + 1))

    # Other shortcuts
    baselines_dir = os.path.join(base_dir, 'pca_baselines')

    # -------------------------------------------------------------------------
    # Make plots for different stacking factors
    # -------------------------------------------------------------------------

    # Run for each stacking factor
    for stacking_factor in config['stacking_factors']:

        print(f'Running for stacking factor {stacking_factor}...',
              end=' ', flush=True)

        # Construct path to result dir for this stacking factor
        result_dir = os.path.join(baselines_dir, f'stacked_{stacking_factor}')

        # Read in the CSV file with the figures of merit into a dataframe
        file_path = os.path.join(result_dir, 'figures_of_merit.csv')
        dataframe = pd.read_csv(filepath_or_buffer=file_path, sep='\t',
                                header=[0, 1], index_col=[0, 1])

        # Set up a figure
        plt.figure(figsize=tuple(plot_size))

        # Make plot for each planet individually
        for i, planet_key in enumerate(planet_positions.keys()):
    
            # Construct full name of the planet (target star + letter)
            planet_name = f'{config["metadata"]["TARGET_STAR"]} {planet_key}'

            # Select the SNR values from the dataframe
            snr_values = dataframe[planet_key]['snr'].values

            # Get maximum SNR and draw it separately
            max_snr_idx = int(np.argmax(snr_values))
            plt.plot(pc_numbers[max_snr_idx], snr_values[max_snr_idx],
                     color=adjust_luminosity(f'C{i}'), marker='x', ms=10)

            # Plot the SNR as a step function
            plt.step(pc_numbers, snr_values, 'o-', color=f'C{i}',
                     where='mid', label=planet_name)

            # Add a label for each data point
            for (n, snr) in zip(pc_numbers, snr_values):
                plt.annotate(s=f'{snr:.2f}\n{n:d}', xy=(n, snr),
                             ha='center', va='center', fontsize=1.5)

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
        plt.xticks(list(range(min_n_components - 1, max_n_components + 1, 5)))
        plt.grid(which='both', ls='--', color='LightGray', alpha=0.5)

        # Save plot as a PDF
        file_path = os.path.join(result_dir, 'snr_over_npc.pdf')
        plt.savefig(file_path, bbox_inches='tight', pad=0)
        plt.clf()

        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
