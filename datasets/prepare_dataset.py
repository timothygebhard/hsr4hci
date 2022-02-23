"""
Run a pre-processing pipeline that does the following things:

1. Extract the stack, parallactic angles, and PSF template from the
   provided PynPoint data base(s) and copy them into a new HDF files.
2. Add metadata, such as the instrument, exposure times, etc. (from the
   configuration file).
3. Add information about position and contrast of planets (from the
   configuration file).
4. Download the headers of the original FITS files which contain
   additional information that are not available from the PynPoint
   data base.
5. Compute a timestamp (in UTC) for every frames.
6. Download observing conditions from the ESO archive and interpolate
   them to match the temporal resolution of the frames.
7. Interpolate and add observing conditions that are only available
   from the original FITS file headers (e.g., mirror temperature).
8. Create plots of the observing conditions.
9. Run PCA (with a fixed number of components) and create a plot to
   sanity-check the planet positions.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from warnings import warn

import argparse
import json
import time

from astropy.units import Quantity
from astroquery.eso import Eso
from photutils import CircularAperture

import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from hsr4hci.coordinates import get_center, polar2cartesian
from hsr4hci.general import prestack_array
from hsr4hci.hdf import save_dict_to_hdf, save_data_to_hdf
from hsr4hci.observing_conditions import get_observing_conditions
from hsr4hci.pca import get_pca_signal_estimates
from hsr4hci.plotting import zerocenter_imshow, disable_ticks
from hsr4hci.time_conversion import (
    date_string_to_timestamp,
    timestamp_to_datetime,
)
from hsr4hci.units import InstrumentUnitsContext


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nPREPARE DATASET\n', flush=True)

    # -------------------------------------------------------------------------
    # Parse command line arguments and load configuration file for data set
    # -------------------------------------------------------------------------

    # Set up a parser and parse the command line arguments to get the data set
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--dataset',
        type=str,
        metavar='NAME',
        required=True,
        help='The name of the data set for which to run the preparation.',
    )
    item = arg_parser.parse_args().dataset

    # Load the configuration file for this data set
    print('Loading data set configuration file...', end=' ', flush=True)
    file_path = Path('.') / item / f'{item}.json'
    with open(file_path, 'r') as json_file:
        config = json.load(json_file)
    print('Done!\n')

    # -------------------------------------------------------------------------
    # Prepare new output HDF file to which all data will be written
    # -------------------------------------------------------------------------

    # Define shortcuts for input and output directories
    input_dir = Path('.') / item / 'input'
    output_dir = Path('.') / item / 'output'
    output_dir.mkdir(exist_ok=True)
    plots_dir = Path('.') / item / 'plots'
    plots_dir.mkdir(exist_ok=True)
    obscon_dir = plots_dir / 'observing_conditions'
    obscon_dir.mkdir(exist_ok=True)

    # Define path to the output HDF file
    output_file_path = output_dir / f'{item}.hdf'

    # -------------------------------------------------------------------------
    # Store metadata and information about planets
    # -------------------------------------------------------------------------

    # Copy metadata from config to output HDF file
    print('Copying metadata to output file...', end=' ', flush=True)
    save_dict_to_hdf(
        dictionary=config['metadata'],
        file_path=output_file_path,
        prefix='metadata',
    )
    print('Done!')

    # Copy information about planets to output HDF file
    print('Copying planet information to output file...', end=' ', flush=True)
    save_dict_to_hdf(
        dictionary=config['planets'],
        file_path=output_file_path,
        prefix='planets',
    )
    print('Done!\n')

    # -------------------------------------------------------------------------
    # Copy data and PynPoint headers from input to output HDF file
    # -------------------------------------------------------------------------

    # Copy the stack, the parallactic angles, and the PSF template
    with h5py.File(output_file_path, 'a') as output_file:
        for item in ('stack', 'parang', 'psf_template'):

            # Copy the data set from the input to the output file. This should
            # preserve the data set's attributes (e.g., the PynPoint history).
            file_path = input_dir / config['input_data'][item]['file_name']
            with h5py.File(file_path, 'r') as input_file:

                # Check if the item already exists in the output file, in
                # which case we need to delete it before we can copy it
                if item in output_file:
                    del output_file[item]

                # Copy the stack / parang / psf_template to the output file
                print(f'Copying /{item}...', end=' ', flush=True)
                input_file.copy(
                    source=config['input_data'][item]['key'],
                    dest=output_file,
                    name=item,
                )
                print('Done!')

                # For the stack and the PSF template, we also copy the
                # PynPoint headers, which contains additional information
                # about the original files, the pre-processing, ...
                if item in ('stack', 'psf_template'):

                    # Define paths to the header in the input and output file
                    header_key = 'header_' + config['input_data'][item]['key']
                    header_name = 'header_' + item

                    # Check if the header already exists in the output file,
                    # in which case we need to delete it before we can copy it
                    if header_name in output_file:
                        del output_file[header_name]

                    # Copy the header (if it exists)
                    print(f'Copying /{header_name}...', end=' ', flush=True)
                    if header_key in input_file:
                        input_file.copy(
                            source=header_key,
                            dest=output_file,
                            name=header_name,
                        )
                        print('Done!')
                    else:
                        print('Failed!')
                        warn(f'No HEADER information found for {item}!')

    # -------------------------------------------------------------------------
    # Download FITS headers and store in CSV / HDF
    # -------------------------------------------------------------------------

    # Read in file with the product IDs, that is, the "proper" names of the
    # FITS files on the ESO archive (e.g., "NACO.2013-02-01T04:42:52.013")
    file_path = input_dir / 'product_ids.txt'
    with open(file_path, 'r') as txt_file:
        product_ids = [_.strip() for _ in txt_file.readlines()]

    # Use astroquery to download the headers for all FITS files and store
    # them in a pandas data frame (this may take a while...), which we sort
    # by the original instrument filename (e.g., "NACO_CORO_SCI032_0014")
    print('\nDownloading FITS headers...', end=' ', flush=True)
    headers = Eso.get_headers(product_ids=product_ids).to_pandas()
    headers.sort_values(by=['HIERARCH ESO DET EXP NAME'], inplace=True)
    print('Done!', flush=True)

    # Store the FITS headers as a CSV file
    print('Storing FITS headers as a CSV file...', end=' ', flush=True)
    file_path = output_dir / 'fits_headers.csv'
    headers.to_csv(file_path, sep='\t')
    print('Done!', flush=True)

    # Add the full FITS headers to the output HDF file
    print('Storing FITS headers in HDF file...', end=' ', flush=True)
    with h5py.File(output_file_path, 'a') as output_file:
        for key in headers.keys():
            save_data_to_hdf(
                hdf_file=output_file,
                location='fits_headers',
                name=key,
                data=headers[key].values,
            )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Collect quantities that we need for the timestamp computation
    # -------------------------------------------------------------------------

    # Find the file names of the FITS files used for the stack (without the
    # ".fits" ending), the index that stores all the information about the
    # frame selection, and NDIT (i.e., the "target" number of frames per cube)
    with h5py.File(output_file_path, 'r') as output_file:
        files = np.array(
            [
                _.decode('utf-8').split('/')[-1][:-5]
                for _ in output_file['header_stack']['FILES']
            ]
        )
        index = np.array(output_file['header_stack']['INDEX'])
        n_dit_per_cube = np.array(output_file['header_stack']['NDIT'])

    # Select the subset of the headers data frame that corresponds to the
    # FITS files used for the stack (as a pandas data frame)
    stack_headers = headers[headers['HIERARCH ESO DET EXP NAME'].isin(files)]
    if stack_headers.empty:
        stack_headers = headers[headers['DP.ID'].isin(files)]
    if stack_headers.empty:
        raise RuntimeError('Could not match stack cubes with their headers!')

    # Get the NAXIS3 values, which is the actual number of frames per cube
    # *before* any frame selection takes place. Because of frame loss, this
    # number can be smaller than NDIT. If *no* frame loss occurred, however,
    # we should have NAXIS3 = NDIT + 1, because the last frame of every cube
    # is the mean of the entire cube.
    n_axis_3 = stack_headers['NAXIS3'].values

    # Compute the number of frames per cube *after* frame selection. Usually,
    # this exists in the PynPoint headers, except for when it does not...
    n_frames_per_cube = np.array([]).astype(int)
    positions = [0] + list(np.cumsum(n_axis_3))
    for start, end in zip(positions[:-1], positions[1:]):
        n_frames_per_cube = np.append(
            n_frames_per_cube, int(np.sum((start <= index) & (index < end)))
        )

    # -------------------------------------------------------------------------
    # Compute the UTC timestamp for every frame
    # -------------------------------------------------------------------------

    print('\nComputing timestamp for each frame...', end=' ', flush=True)

    with h5py.File(output_file_path, 'a') as output_file:

        # Get the number of frames in the stack
        cube_dates = output_file['header_stack']['DATE']
        n_cubes = len(cube_dates)
        dit = output_file['stack'].attrs['DIT']
        parang_shape = output_file['parang'].shape

        # Keep track of the timestamps of all frames
        timestamps_utc_list = []

        # Compute the start and end of every cube as UTC timestamps
        cube_starts = [date_string_to_timestamp(_) for _ in cube_dates]
        cube_ends = [
            start + n_dit * dit
            for start, n_dit in zip(cube_starts, n_dit_per_cube)
        ]

        # Loop over cubes and compute timestamps for frames in cube
        for i in range(n_cubes):

            # Define shortcuts for values of this cube
            cube_start = cube_starts[i]
            cube_end = cube_ends[i]
            n_frames = n_frames_per_cube[i]
            n_dit = n_dit_per_cube[i]

            # Now we compute a timestamp for every frame in the cube. As long
            # as we do not have frame loss (i.e., NAXIS3 == NDIT + 1; the + 1
            # is because the last frame in a cube is, by default, the mean of
            # the cube), we can find the exact time for each frame.
            if n_axis_3[i] == n_dit + 1:

                # Compute timestamps for all frames in the cube (including
                # those that were removed by the frame selection)
                timestamps = np.linspace(cube_start, cube_end, n_dit)

                # Get the selection mask for the
                mask = np.full(np.sum(n_axis_3), False)
                mask[index] = True
                start = ([0] + list(np.cumsum(n_axis_3)))[i]
                mask = mask[start : start + n_dit]

                # Finally, select only the timestamps for the selected frames
                timestamps_utc_list += list(timestamps[mask])

            # If we have lost frames (e.g., due to problems with the read-out),
            # we need to improvise, because there is no way of knowing *which*
            # got lost: In this case, we simply distribute the frames that we
            # have uniformly over the duration of the cube.
            else:
                timestamps_utc_list += list(
                    np.linspace(cube_start, cube_end, n_frames)
                )

        # Convert timestamps to a numpy array
        timestamps_utc = np.array(timestamps_utc_list)

        # Save the result (i.e., the timestamps for all frames in all cubes)
        output_file.require_dataset(
            name='timestamps_utc', shape=parang_shape, dtype=float
        )
        output_file['timestamps_utc'][...] = timestamps_utc

    print('Done!\n')

    # -------------------------------------------------------------------------
    # Query, interpolate and add observing conditions (from archives)
    # -------------------------------------------------------------------------

    with h5py.File(output_file_path, 'a') as output_file:

        # Get the start and end times of all cubes as datetime objects:
        # We will add the time spans of the cubes to the plots of the
        # parameters as an additional sanity check
        cube_starts_dates = [timestamp_to_datetime(_) for _ in cube_starts]
        cube_ends_dates = [timestamp_to_datetime(_) for _ in cube_ends]

        # Create new data frame in which we store all interpolated observing
        # conditions to create one final correlation plots
        observing_conditions_df = pd.DataFrame()

        for parameter_name in (
            'air_pressure',
            'coherence_time',
            'isoplanatic_angle',
            'observatory_temperature',
            'relative_humidity',
            'seeing',
            'wind_speed_u',
            'wind_speed_v',
            'wind_speed_w',
        ):

            print(f'Retrieving {parameter_name}...', end=' ', flush=True)

            # Query ESO archive and interpolate observing conditions
            try:
                interpolated, query_results = get_observing_conditions(
                    parameter_name=parameter_name, timestamps=timestamps_utc
                )
            except IndexError:
                print('Failed!', flush=True)
                continue

            # Store interpolated observing conditions in data frame
            observing_conditions_df[parameter_name] = interpolated

            # Store interpolated observing conditions in the output HDF file
            save_data_to_hdf(
                hdf_file=output_file,
                location='observing_conditions/interpolated',
                name=parameter_name,
                data=interpolated,
            )

            # Store the query results in the output HDF file
            location = f'observing_conditions/query_results/{parameter_name}'
            for key, value in query_results.items():
                save_data_to_hdf(
                    hdf_file=output_file,
                    location=location,
                    name=key,
                    data=value,
                )

            print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # "Interpolate" and add observing conditions (from FITS headers)
    # -------------------------------------------------------------------------

    # Some of the telescope-specific observing conditions are not available
    # from the central archives, but are only stored in the headers of the
    # FITS files of the observations.

    with h5py.File(output_file_path, 'a') as output_file:

        # Loop over the different parameters that are only available from
        # the FITS headers. For some values (e.g., air mass), there is a
        # START and END value, referring to the beginning and end of the cube.
        # For other values, there is only a single value for the entire cube.
        # Parameters whose name end in '__fits' are only for debugging.
        for parameter_name, parameter_start_key, parameter_end_key, in (
            (
                'air_mass',
                'HIERARCH ESO TEL AIRM START',
                'HIERARCH ESO TEL AIRM END',
            ),
            (
                'm1_temperature',
                'HIERARCH ESO TEL TH M1 TEMP',
                'HIERARCH ESO TEL TH M1 TEMP',
            ),
            (
                'detector_temperature',
                'HIERARCH ESO INS TEMP DET',
                'HIERARCH ESO INS TEMP DET',
            ),
            (
                'seeing__fits',
                'HIERARCH ESO TEL AMBI FWHM START',
                'HIERARCH ESO TEL AMBI FWHM END',
            ),
            (
                'air_pressure__fits',
                'HIERARCH ESO TEL AMBI PRES START',
                'HIERARCH ESO TEL AMBI PRES END',
            ),
            (
                'relative_humidity__fits',
                'HIERARCH ESO TEL AMBI RHUM',
                'HIERARCH ESO TEL AMBI RHUM',
            ),
            (
                'coherence_time__fits',
                'HIERARCH ESO TEL AMBI TAU0',
                'HIERARCH ESO TEL AMBI TAU0',
            ),
            (
                'observatory_temperature__fits',
                'HIERARCH ESO TEL AMBI TEMP',
                'HIERARCH ESO TEL AMBI TEMP',
            ),
            (
                'wind_direction__fits',
                'HIERARCH ESO TEL AMBI WINDDIR',
                'HIERARCH ESO TEL AMBI WINDDIR',
            ),
            (
                'wind_speed__fits',
                'HIERARCH ESO TEL AMBI WINDSP',
                'HIERARCH ESO TEL AMBI WINDSP',
            ),
        ):

            print(f'Retrieving {parameter_name}...', end=' ', flush=True)

            # Keep track of the value of the parameter for each frame
            parameter_values_list = []

            # Loop over cubes and "interpolate" observing conditions from FITS
            for i in range(n_cubes):

                # Get the value at the start and end of the cube
                start_value = stack_headers[parameter_start_key].values[i]
                end_value = stack_headers[parameter_end_key].values[i]

                # Get the number of cube in the frame (actual and target)
                n_frames = n_frames_per_cube[i]
                n_dit = n_dit_per_cube[i]

                # Again, we need to distinguish between "no frame loss":
                if n_axis_3[i] == n_dit + 1:

                    # Linearly interpolate the parameter values over *all*
                    # frames in the cube (including those that were removed
                    # by the frame selection)
                    dummy = np.linspace(start_value, end_value, n_dit)

                    # Get the selection mask for this cube
                    mask = np.full(np.sum(n_axis_3), False)
                    mask[index] = True
                    start = ([0] + list(np.cumsum(n_axis_3)))[i]
                    mask = mask[start : start + n_dit]

                    # Apply the selection mask, that is, only keep the values
                    # that belong to frames kept by the frame selection
                    parameter_values_list += list(dummy[mask])

                # ...and "frame loss":
                else:

                    # In this case, we simply distribute the the frames (or
                    # rather, the parameter values for the frames) uniformly
                    # over the entire cube
                    parameter_values_list += list(
                        np.linspace(start_value, end_value, n_frames)
                    )

            # Convert parameter values to array and store in data frame
            # (unless it is a debugging parameter)
            parameter_values = np.array(parameter_values_list)

            # Store the parameter values in the output HDF file and the data
            # frame (parameters ending in '__fits' are only for debugging)
            if not parameter_name.endswith('__fits'):
                observing_conditions_df[parameter_name] = parameter_values
                save_data_to_hdf(
                    hdf_file=output_file,
                    location='observing_conditions/interpolated',
                    name=parameter_name,
                    data=parameter_values,
                )
            else:
                save_data_to_hdf(
                    hdf_file=output_file,
                    location='observing_conditions/debugging',
                    name=parameter_name,
                    data=parameter_values,
                )

            print('Done!', flush=True)

    print()

    # -------------------------------------------------------------------------
    # Create individual plots for all observing conditions
    # -------------------------------------------------------------------------

    # Convert timestamps to datetime objects (for plotting)
    datetimes = [timestamp_to_datetime(_) for _ in timestamps_utc]

    # Loop over parameters and create plots
    for parameter_name in sorted(observing_conditions_df.keys()):

        print(f'Plotting {parameter_name}...', end=' ', flush=True)

        # Get the parameter values
        parameter_values = observing_conditions_df[parameter_name].values

        # Create a plot of the interpolated parameter and save as a PDF
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(datetimes, parameter_values, '.', ms=1, mew=0.1)
        ax.set_title(parameter_name)
        ax.grid(which='both', ls='--', color='gray', lw=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate()
        for xmin, xmax in zip(cube_starts_dates, cube_ends_dates):
            ax.axvspan(xmin, xmax, facecolor='lightgray', alpha=0.3)
        file_path = obscon_dir / f'{parameter_name}.pdf'
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Plot a correlation matrix
    # -------------------------------------------------------------------------

    print('Plotting correlation matrix...', end=' ', flush=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    correlation_matrix = observing_conditions_df.reindex(
        sorted(observing_conditions_df.columns), axis=1
    ).corr()
    heatmap = sns.heatmap(
        data=correlation_matrix,
        square=True,
        linewidths=2,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        cbar=False,
        annot=True,
        annot_kws={'size': 12},
    )
    ax.set_yticklabels(correlation_matrix.columns, rotation=0)
    ax.set_xticklabels(correlation_matrix.columns)
    plt.tight_layout()
    file_path = obscon_dir / '_correlations.pdf'
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig=fig)

    print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Create of the signal estimate (from PCA) as a general sanity check
    # -------------------------------------------------------------------------

    with h5py.File(output_file_path, 'r') as output_file:

        # Temporally bin the stack and the parallactic angles
        print('Temporally binning stack and parang...', end=' ', flush=True)
        stack = prestack_array(np.array(output_file['stack']), 100)
        parang = prestack_array(np.array(output_file['parang']), 100)
        print('Done!\n', flush=True)

        # Run PCA with 20 components on temporally binned version of stack
        print('Computing PCA-based signal estimate...', end=' ', flush=True)
        signal_estimate = get_pca_signal_estimates(
            stack=stack,
            parang=parang,
            n_components=20,
            return_components=False,
        )
        signal_estimate = np.asarray(signal_estimate).squeeze()
        print('Done!\n', flush=True)

        # Define the unit conversion context for this data set
        pixscale = Quantity(config['metadata']['PIXSCALE'], 'arcsec / pixel')
        lambda_over_d = Quantity(config['metadata']['LAMBDA_OVER_D'], 'arcsec')
        instrument_units_context = InstrumentUnitsContext(
            pixscale=pixscale, lambda_over_d=lambda_over_d
        )

        # Define shortcuts
        frame_size = (signal_estimate.shape[0], signal_estimate.shape[1])
        center = get_center(frame_size)

        # Plot the signal estimate; set up some plot options; add plot title
        print('Plotting signal estimate...', end=' ', flush=True)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(signal_estimate, origin='lower', cmap='RdBu_r')
        ax.plot(center[0], center[1], '+', color='black')
        zerocenter_imshow(ax)
        disable_ticks(ax)
        fig.suptitle(
            f'{config["metadata"]["TARGET_STAR"]} (temporally binned, 20 PCs)'
        )
        fig.tight_layout()

        # Add the planet positions to the plot
        with instrument_units_context:
            for planet_name, planet_params in config['planets'].items():
                position = polar2cartesian(
                    separation=Quantity(planet_params['separation'], 'arcsec'),
                    angle=Quantity(planet_params['position_angle'], 'degree'),
                    frame_size=frame_size,
                )
                aperture = CircularAperture(
                    positions=position,
                    r=lambda_over_d.to('pixel').value,
                )
                # noinspection PyTypeChecker
                aperture.plot(axes=ax, color='black', ls='--')
                ax.annotate(
                    text=planet_name,
                    xy=position,
                    ha='center',
                    va='center',
                    xytext=(20, 0),
                    textcoords='offset pixels',
                )
        file_path = plots_dir / 'signal_estimate.pdf'
        plt.savefig(file_path, bbox_inches='tight')
        print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
