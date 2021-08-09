"""
Create plot to compare the expected signal with an observing condition.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from argparse import ArgumentParser

import time

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import numpy as np

from hsr4hci.data import (
    load_psf_template,
    load_parang,
    load_observing_conditions,
)
from hsr4hci.forward_modeling import get_time_series_for_position


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
    parser.add_argument('--dataset', type=str, default='beta_pictoris__lp')
    parser.add_argument('--binning-factor', type=int, default=1)
    parser.add_argument('--signal-time', type=int, default=19640)
    parser.add_argument('--x', type=int, default=30)
    parser.add_argument('--y', type=int, default=30)
    parser.add_argument(
        '--observing-condition', type=str, default='m1_temperature'
    )

    # Define shortcuts
    args = parser.parse_args()
    dataset = args.dataset
    binning_factor = args.binning_factor
    signal_time = args.signal_time
    x = args.x
    y = args.y
    observing_condition = args.observing_condition

    # -------------------------------------------------------------------------
    # Load data without planets, add fake planet
    # -------------------------------------------------------------------------

    # Load data without planet
    print('Loading data...', end=' ', flush=True)
    parang = load_parang(name_or_path=dataset, binning_factor=binning_factor)
    psf_template = load_psf_template(name_or_path=dataset)
    observing_conditions = load_observing_conditions(
        name_or_path=dataset, binning_factor=binning_factor
    )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Load data without planets, add fake planet
    # -------------------------------------------------------------------------

    # Compute expected signal time series for the position and normalize it
    expected_signal = get_time_series_for_position(
        position=(x, y),
        signal_time=signal_time,
        frame_size=(65, 65),
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
    # observing condition
    correlation = np.corrcoef(expected_signal, array)[0, 1]
    print('Correlation:', correlation)

    # Fit a linear model: basically test how well we can express the expected
    # signal by the observing condition
    model = LinearRegression().fit(X=array.reshape(-1, 1), y=expected_signal)
    prediction = model.predict(array.reshape(-1, 1))

    # Create plot
    fig, ax1 = plt.subplots(figsize=(7.24551 / 2, 7.24551 / 4))
    ax2 = ax1.twinx()
    ax2.set_ylim(-2.5, 2.5)

    ax1.plot(
        observing_conditions.as_dataframe()[observing_condition].values,
        color='C0',
        label=observing_condition,
    )
    ax1.tick_params(axis='y', labelcolor='C0')
    ax1.set_ylabel('Temperature (°C)', color='C0')

    ax2.plot(expected_signal, color='k', lw=3, label='Expected planet signal')
    ax2.plot(prediction, color='C1', label=f'{observing_condition} (scaled)')
    ax2.tick_params(axis='y', labelcolor='C1')
    ax2.set_ylabel('Parameter value (arbitrary units)', color='C1')

    # Set plot options
    ax2.set_xlabel('Time (frame index)')
    ax2.legend(loc='best', fontsize=6)

    # Set up font sizes
    for ax in (ax1, ax2):
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(6)

    # Save plot as a PDF
    plt.savefig(
        f'{observing_condition}.pdf',
        dpi=600,
        bbox_inches='tight',
        pad_inches=0,
    )

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
