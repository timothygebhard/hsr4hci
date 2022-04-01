"""
This script will plot the interpolated observing conditions and compare
them with the raw values from the ESO archive and the values from the
FITS headers. By default, it will look at the air pressure for the
beta_pictoris__lp data set.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from datetime import timedelta

import time

from matplotlib.dates import MinuteLocator, SecondLocator, DateFormatter

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hsr4hci.config import get_datasets_dir
from hsr4hci.plotting import set_fontsize
from hsr4hci.time_conversion import (
    date_string_to_timestamp,
    round_minutes,
    timestamp_to_datetime,
)


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
    # Load the data from the HDF file
    # -------------------------------------------------------------------------

    # Define path to data set file
    dataset = 'beta_pictoris__lp'
    file_path = get_datasets_dir() / dataset / 'output' / f'{dataset}.hdf'

    with h5py.File(file_path, 'r') as hdf_file:

        # Get the UTC timestamps of all frames
        timestamps = np.array(hdf_file['timestamps_utc'])

        # Load the headers and convert them into a pandas DataFrame
        headers_dict = {}
        for key in hdf_file['fits_headers'].keys():
            headers_dict[key] = np.array(hdf_file['fits_headers'][key])
        headers = pd.DataFrame(headers_dict)

        # Decode the file frame (from b'...' to string) and sort data frame
        headers['HIERARCH ESO DET EXP NAME'] = headers[
            'HIERARCH ESO DET EXP NAME'
        ].apply(lambda x: x.decode("utf-8"))
        headers.sort_values(by=['HIERARCH ESO DET EXP NAME'], inplace=True)

        # Get the names of the original FITS files as strings
        files = np.array(
            [
                _.decode('utf-8').split('/')[-1][:-5]
                for _ in hdf_file['header_stack']['FILES']
            ]
        )

        # Select the headers of the files that were used for the stack (i.e.,
        # exclude the headers of calibration files)
        stack_headers = headers[
            headers['HIERARCH ESO DET EXP NAME'].isin(files)
        ]

        # Compute the start and end of every cube as UTC timestamps
        cube_dates = hdf_file['header_stack']['DATE']
        dit = hdf_file['stack'].attrs['DIT']
        n_dit_per_cube = np.array(hdf_file['header_stack']['NDIT'])
        cube_starts = [date_string_to_timestamp(_) for _ in cube_dates]
        cube_ends = [
            start + n_dit * dit
            for start, n_dit in zip(cube_starts, n_dit_per_cube)
        ]

        # Get the interpolated air pressure values, as well as the query
        # results (i.e., the raw values from the ESO ambient server)
        group = hdf_file['observing_conditions']
        interpolated = np.array(group['interpolated']['air_pressure'])
        query_results = dict(
            integration_time=np.array(
                group['query_results']['air_pressure']['integration_time']
            ),
            parameter=np.array(
                group['query_results']['air_pressure']['parameter']
            ),
            timestamp=np.array(
                group['query_results']['air_pressure']['timestamp']
            ),
        )

        # Get the air pressure values (start and end) for each cube
        start_values = stack_headers['HIERARCH ESO TEL AMBI PRES START'].values
        end_values = stack_headers['HIERARCH ESO TEL AMBI PRES END'].values

    # -------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------

    # Create new figure
    fig, ax = plt.subplots(figsize=(18.4 / 2.54, 4 / 2.54))
    fig.subplots_adjust(left=0.07, right=0.999, top=0.9, bottom=0.32)

    # Set fontsize
    set_fontsize(ax=ax, fontsize=6)

    # Plot the cubes as shaded regions in the background
    cube_starts_dates = [timestamp_to_datetime(_) for _ in cube_starts]
    cube_ends_dates = [timestamp_to_datetime(_) for _ in cube_ends]
    for i, (xmin, xmax) in enumerate(zip(cube_starts_dates, cube_ends_dates)):
        label = 'Cubes' if i == 0 else None
        ax.axvspan(xmin, xmax, facecolor='lightgray', alpha=0.5, label=label)

    # Plot the start and end values (from FITS) for each cube
    for i, (start_time, end_time, start_value, end_value) in enumerate(
        zip(cube_starts_dates, cube_ends_dates, start_values, end_values)
    ):
        label = 'Values from FITS headers' if i == 0 else None
        ax.plot(
            [start_time, end_time],
            [start_value, end_value],
            '-',
            label=label,
            color='gray',
        )
        ax.plot(
            start_time,
            start_value,
            'd',
            ms=2,
            color='gray',
        )
        ax.plot(
            end_time,
            end_value,
            'd',
            ms=2,
            color='gray',
        )

    # Plot the values queried from the ambient servers
    for i, (timestamp, integration_time, value) in enumerate(
        zip(
            query_results['timestamp'],
            query_results['integration_time'],
            query_results['parameter'],
        )
    ):
        label = 'Values from METEO archive' if i == 0 else None
        start_time = timestamp_to_datetime(timestamp - integration_time)
        end_time = timestamp_to_datetime(timestamp)
        ax.plot(
            [start_time, end_time],
            [value, value],
            '-',
            color='C1',
            lw=2,
            label=label,
        )
        ax.plot(
            end_time,
            value,
            's',
            color='C1',
            ms=4,
        )

    # Plot the interpolated values
    datetimes = [timestamp_to_datetime(_) for _ in timestamps]
    ax.plot(
        [],
        [],
        lw=2,
        color='C0',
        label='Interpolated values',
    )
    ax.plot(
        datetimes,
        interpolated,
        '.',
        ms=5,
        mew=0,
    )

    # Set axes limits
    xmin_ = round_minutes(datetimes[0] + timedelta(seconds=740), 'down', 1)
    xmax_ = round_minutes(xmin_ + timedelta(seconds=560), 'up ', 1)
    ax.set_xlim(xmin_, xmax_)
    ax.set_ylim(743.2, 743.5)

    # Set up formatting for x-axis (i.e., use properly formatted dates / times)
    ax.xaxis.set_major_locator(MinuteLocator())
    ax.xaxis.set_minor_locator(SecondLocator(range(0, 60, 10)))
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    for tick_label in ax.get_xticklabels():
        tick_label.set_ha('right')
        tick_label.set_rotation(30)

    # Add axes labels
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Air pressure (hPa)')

    # Add legend
    ax.legend(
        fontsize=6,
        ncol=4,
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        mode="expand",
        borderaxespad=0.0,
        frameon=False,
    )

    # Save the result as a PDF
    plt.savefig('air_pressure.pdf', dpi=600)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
