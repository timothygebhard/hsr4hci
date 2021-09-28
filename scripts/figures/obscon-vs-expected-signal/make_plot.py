"""
Create plot to compare the expected signal with an observing condition.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from argparse import ArgumentParser

import time

from matplotlib.dates import HourLocator, MinuteLocator, DateFormatter
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression

import h5py
import matplotlib.pyplot as plt
import numpy as np

from hsr4hci.config import get_datasets_dir
from hsr4hci.data import (
    load_psf_template,
    load_parang,
    load_observing_conditions,
)
from hsr4hci.forward_modeling import get_time_series_for_position
from hsr4hci.plotting import set_fontsize
from hsr4hci.time_conversion import timestamp_to_datetime, round_minutes


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nMAKE PLOT\n', flush=True)

    # -------------------------------------------------------------------------
    # Parse command line arguments
    # -------------------------------------------------------------------------

    # Set up a parser and parse the command line arguments
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='r_cra__lp')
    parser.add_argument('--binning-factor', type=int, default=1)
    parser.add_argument('--relative-signal-time', type=int, default=0.9677)
    parser.add_argument('--x', type=int, default=30)
    parser.add_argument('--y', type=int, default=33)
    parser.add_argument(
        '--observing-condition', type=str, default='wind_speed_u'
    )
    parser.add_argument('--y-label', type=str, default='Wind speed $U$')
    parser.add_argument('--y-unit', type=str, default='(m/s)')

    # Define shortcuts
    args = parser.parse_args()
    dataset = args.dataset
    binning_factor = args.binning_factor
    relative_signal_time = args.relative_signal_time
    x = args.x
    y = args.y
    observing_condition = args.observing_condition
    y_label = args.y_label
    y_unit = args.y_unit

    # -------------------------------------------------------------------------
    # Load data without planets, add fake planet
    # -------------------------------------------------------------------------

    # Load data without planet
    print('Loading data...', end=' ', flush=True)
    parang = load_parang(name_or_path=dataset, binning_factor=binning_factor)
    psf_template = load_psf_template(name_or_path=dataset)
    observing_conditions = load_observing_conditions(
        name_or_path=dataset,
        binning_factor=binning_factor,
    )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Prepare plot: get grid, compute expected signal, compute correlation, ...
    # -------------------------------------------------------------------------

    # Load timestamps and define grid for plotting
    file_path = get_datasets_dir() / dataset / 'output' / f'{dataset}.hdf'
    with h5py.File(file_path, 'r') as hdf_file:
        grid = np.array(
            [timestamp_to_datetime(_) for _ in hdf_file['timestamps_utc']]
        )

    # Compute absolute signal time
    signal_time = int(relative_signal_time * len(parang) - 1)

    # Compute expected signal time series for the position and normalize it
    expected_signal = get_time_series_for_position(
        position=(x, y),
        signal_time=signal_time,
        frame_size=(51, 51),
        parang=parang,
        psf_template=psf_template,
    )
    expected_signal -= np.mean(expected_signal)
    expected_signal /= np.std(expected_signal)

    # Select and normalize the observing condition
    array = observing_conditions.as_dataframe()[observing_condition].values
    array -= np.mean(array)
    array /= np.std(array)

    # Compute the correlation coefficient between the expected signal and the
    # normalized observing condition
    correlation = np.corrcoef(expected_signal, array)[0, 1]
    print('Correlation:', correlation)

    # Fit a linear model: basically test how well we can express the expected
    # signal by the observing condition
    model = LinearRegression().fit(X=array.reshape(-1, 1), y=expected_signal)
    prediction = model.predict(array.reshape(-1, 1))

    # -------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------

    # Create plot
    fig, ax1 = plt.subplots(figsize=(9 / 2.54, 4.5 / 2.54))
    fig.subplots_adjust(left=0.11, right=0.90, top=0.99, bottom=0.5)
    ax2 = ax1.twinx()

    # Plot raw observing condition, expected signal, and best fit
    ax1.plot(
        grid,
        observing_conditions.as_dataframe()[observing_condition].values,
        color='C0',
        lw=2,
    )
    ax2.plot(grid, expected_signal, color='k', lw=2)
    ax2.plot(grid, prediction, color='C1', lw=2)

    # -------------------------------------------------------------------------
    # Setup plot options
    # -------------------------------------------------------------------------

    # Add limits for the axes
    ax1.set_xlim(round_minutes(grid[0], 'down'), round_minutes(grid[-1], 'up'))
    ax2.set_ylim(-3, 3)

    # Set up font sizes
    for ax in (ax1, ax2):
        set_fontsize(ax=ax, fontsize=6)

    # Format the x-axis as dates
    ax1.xaxis.set_major_locator(HourLocator())
    ax1.xaxis.set_minor_locator(MinuteLocator(range(0, 60, 10)))
    ax1.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate(rotation=0, ha='center')
    ax1.set_xlabel('Time (UTC)')

    # Set y-axis labels and define colors of labels and ticks
    ax1.set_ylabel(f'{y_label} {y_unit}', color='C0')
    ax2.set_ylabel(
        f'Scaled {y_label[0].lower()}{y_label[1:]} (unitless)', color='C1'
    )
    ax1.tick_params(axis='y', labelcolor='C0')
    ax2.tick_params(axis='y', labelcolor='C1')

    # Add a grid (only for the x-axis)
    ax1.grid(
        b=True,
        which='both',
        axis='x',
        lw=1,
        alpha=0.3,
        dash_capstyle='round',
        dashes=(0, 2),
    )

    # -------------------------------------------------------------------------
    # Add a legend and save the plot as a PDF
    # -------------------------------------------------------------------------

    # Manually add a legend to to plot
    handles = [
        Line2D([0], [0], color='black', lw=2),
        Line2D([0], [0], color='C0', lw=2),
        Line2D([0], [0], color='C1', lw=2),
    ]
    labels = [
        "Expected planet signal",
        f"{y_label} (raw)",
        f"{y_label} (scaled)",
    ]
    ax1.legend(handles=handles, labels=labels, fontsize=6)

    # Save plot as a PDF
    plt.savefig(f'{observing_condition}.pdf', dpi=600)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
