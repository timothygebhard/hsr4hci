"""
This script can be used to create various plots of the observing
condition parameters obtained with 01_get_observing_conditions.py.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from textwrap import wrap

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from hsr4hci.utils.argparsing import get_base_directory
from hsr4hci.utils.observing_conditions import get_description_and_unit, \
    load_observing_conditions


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nMAKE PLOTS FOR OBSERVING CONDITIONS\n', flush=True)

    # -------------------------------------------------------------------------
    # Parse command line arguments and load observing conditions
    # -------------------------------------------------------------------------

    # Get base_directory from command line arguments
    base_dir = get_base_directory()

    # Construct path to result dir and plot dir; make sure the latter exists
    results_dir = os.path.join(base_dir, 'observing_conditions')
    plots_dir = os.path.join(results_dir, 'plots')
    Path(plots_dir).mkdir(exist_ok=True)

    # Load observing conditions as a pandas DataFrame
    print('Loading observing conditions...', end=' ', flush=True)
    file_path = os.path.join(results_dir, 'observing_conditions.hdf')
    observing_conditions: pd.DataFrame = \
        load_observing_conditions(file_path=file_path,
                                  as_dataframe=True)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Plot the value of each parameter as a function of time
    # -------------------------------------------------------------------------

    print('Plotting parameter values over time...', end=' ', flush=True)

    # Make sure the respective sub-directory in the plots directory exists
    parameters_dir = os.path.join(plots_dir, 'parameters')
    Path(parameters_dir).mkdir(exist_ok=True)

    # Loop over all observing condition parameters
    for i, parameter in enumerate(observing_conditions):

        # Set up a new figure
        fig, ax = plt.subplots(figsize=(5, 5))

        # Select the values of this parameter and get its unit and description
        values = observing_conditions[parameter]
        description, unit = get_description_and_unit(parameter)

        # Plot values
        ax.plot(values)

        # Use textwrap module to limit line length of descriptions
        description = '\n'.join(wrap(str(description), width=28, max_lines=2))

        # Set up title and axes labels
        ax.set_title(description, wrap=True)
        ax.set_xlabel('Time (in frames)')
        if unit is not None:
            ax.set_ylabel(f'{description} (in {unit})')
        else:
            ax.set_ylabel(description)

        # Set up limits for the x-axis
        ax.set_xlim(0, len(values))

        # Add a grid
        ax.grid(which='both', ls='--', color='lightgray')

        # Save plot and close figure
        plt.tight_layout()
        file_path = os.path.join(parameters_dir, f'{parameter}__values.pdf')
        plt.savefig(file_path, bbox_inches='tight', pad=0)
        plt.close(fig=fig)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Plot the distribution of each parameter as a histogram
    # -------------------------------------------------------------------------

    print('Plotting parameter values histograms...', end=' ', flush=True)

    # Make sure the respective sub-directory in the plots directory exists
    histograms_dir = os.path.join(plots_dir, 'histograms')
    Path(histograms_dir).mkdir(exist_ok=True)

    # Loop over all observing condition parameters
    for i, parameter in enumerate(observing_conditions):

        # Set up a new figure
        fig, ax = plt.subplots(figsize=(5, 5))

        # Select the values of this parameter and get its unit and description
        values = observing_conditions[parameter]
        description, unit = get_description_and_unit(parameter)

        # Use textwrap module to limit line length of descriptions
        description = '\n'.join(wrap(str(description), width=28, max_lines=2))

        # Prepare bins and axis limits
        if parameter == 'relative_humidity':
            bins = np.linspace(0, 100, 32)
            vmin, vmax = -1.0, 101.0
        elif parameter in ('cos_wind_direction', 'sin_wind_direction'):
            bins = np.linspace(-1, 1, 32)
            vmin, vmax = -1.1, 1.1
        else:
            bins = 32
            vmin, vmax = 0, 0

        # Plot the histogram
        _, hist_bins, __ = ax.hist(values, bins=32, histtype='stepfilled')

        # Plot the mean, median and standard deviation
        ax.axvline(x=np.mean(values), label='mean', color='C1')
        ax.axvline(x=np.mean(values) - np.std(values), label='mean ± std',
                   color='C1', ls='--', alpha=0.5)
        ax.axvline(x=np.mean(values) + np.std(values),
                   color='C1', ls='--', alpha=0.5)
        ax.axvline(x=np.median(values), label='median', color='C2')

        # Set up title, axes labels and legend
        ax.set_title(description)
        ax.set_xlabel(description)
        ax.set_ylabel('Frequency')
        ax.legend(loc='best')

        # Set up limits for the x-axis
        if vmin == 0 and vmax == 0:
            vmin = min(hist_bins) - (hist_bins[1] - hist_bins[0])
            vmax = max(hist_bins) + (hist_bins[1] - hist_bins[0])
        ax.set_xlim(vmin, vmax)

        # Save plot and close figure
        plt.tight_layout()
        file_path = os.path.join(histograms_dir, f'{parameter}__histogram.pdf')
        plt.savefig(file_path, bbox_inches='tight', pad=0)
        plt.close(fig=fig)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Use seaborn to create a correlation matrix plot
    # -------------------------------------------------------------------------

    print('Plotting correlation matrix...', end=' ', flush=True)

    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Compute correlation matrix for the parameters of the observing conditions
    correlation_matrix = observing_conditions.corr()

    # Create an annotated heatmap-plot of the correlation matrix
    heatmap = sns.heatmap(data=correlation_matrix,
                          square=True,
                          linewidths=2,
                          cmap='coolwarm',
                          vmin=-1,
                          vmax=1,
                          cbar=False,
                          annot=True,
                          annot_kws={'size': 12})

    # Add the column names as labels
    ax.set_yticklabels(correlation_matrix.columns, rotation=0)
    ax.set_xticklabels(correlation_matrix.columns)

    # Save plot and close figure
    plt.tight_layout()
    file_path = os.path.join(plots_dir, 'correlations.pdf')
    plt.savefig(file_path, bbox_inches='tight', pad=0)
    plt.close(fig=fig)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
