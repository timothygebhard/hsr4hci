"""
This script creates PDF versions of the signal estimates with the
highest SNR, both for PCA and for median ADI.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from ast import literal_eval

import json
import time

from astropy.units import Quantity
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hsr4hci.utils.argparsing import get_base_directory
from hsr4hci.utils.fits import read_fits
from hsr4hci.utils.plotting import plot_frame
from hsr4hci.utils.units import set_units_for_instrument


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCREATING RESULT PLOTS AS PDF AND PNG\n', flush=True)

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
    # Define shortcuts and set up unit conversions
    # -------------------------------------------------------------------------

    # Shortcuts to PIXSCALE and LAMBDA_OVER_D
    metadata = config['metadata']
    pixscale = metadata['PIXSCALE']
    lambda_over_d = metadata['LAMBDA_OVER_D']

    # Use this to set up the instrument-specific conversion factors. We need
    # this here to that we can parse "lambda_over_d" as a unit in the config.
    set_units_for_instrument(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
    )

    # Other shortcuts to elements in the config
    stacking_factors = config['stacking_factors']
    planet_names = list(config['evaluation']['planets'].keys())
    aperture_radius = config['evaluation']['snr_options']['aperture_radius']
    aperture_radius = Quantity(*aperture_radius).to('pixel').value

    # -------------------------------------------------------------------------
    # Loop over stacking factors and create plots
    # -------------------------------------------------------------------------

    # Prepare results directory
    plots_dir = base_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    print('Creating plots:')
    for stacking_factor in tqdm(stacking_factors, ncols=80):

        # ---------------------------------------------------------------------
        # Create plots for median ADI results
        # ---------------------------------------------------------------------

        # Define directory that contains results for this stacking factor
        stacking_dir = (
            base_dir / 'median_adi_baselines' / f'stacked__{stacking_factor}'
        )

        # Collect optimized positions and SNRs for all planets
        positions, snrs = list(), list()
        for planet_name in planet_names:
            file_path = stacking_dir / f'{planet_name}.csv'
            dataframe = pd.read_csv(file_path, header=None).set_index(0).T
            positions.append(literal_eval(dataframe['new_position'].values[0]))
            snrs.append(float(dataframe['snr'].values[0]))

        # Read in the frame containing the signal estimate
        file_path = stacking_dir / 'signal_estimate.fits'
        frame = read_fits(file_path=file_path)

        # Plot the result and save it as a PDF
        file_path = plots_dir / f'madi__{stacking_factor}.pdf'
        plot_frame(
            frame=frame,
            file_path=file_path,
            positions=positions,
            aperture_radius=aperture_radius,
            snrs=snrs,
        )

        # ---------------------------------------------------------------------
        # Create plots for PCA ADI results
        # ---------------------------------------------------------------------

        # Define directory that contains results for this stacking factor
        stacking_dir = (
            base_dir / 'pca_baselines' / f'stacked_{stacking_factor}'
        )

        # Read in the CSV file with the figures of merit into a dataframe
        file_path = stacking_dir / 'figures_of_merit.csv'
        dataframe = pd.read_csv(
            filepath_or_buffer=file_path,
            sep='\t',
            header=[0, 1],
            index_col=[0, 1],
        )

        # Read the stack of signal estimates
        file_path = stacking_dir / 'signal_estimates.fits'
        signal_estimates = read_fits(file_path=file_path)

        # Loop over planets: We need to create each plot multiple times,
        # because different planets will have their maximum SNR for different
        # numbers of principal components
        for planet_name in planet_names:

            # Select the SNR values from the dataframe and get the index of
            # the row that corresponds to the maximum SNR
            snr_values = dataframe[planet_name]['snr'].values
            n_pc = int(np.argmax(snr_values))

            # Collect optimized positions and SNRs for all planets for this
            # number of principal components
            positions, snrs = list(), list()
            for pn in planet_names:
                positions.append(
                    literal_eval(dataframe[pn]['new_position'].values[n_pc])
                )
                snrs.append(dataframe[pn]['snr'].values[n_pc])

            # Select the signal estimate that matches the given number of PCs
            frame = signal_estimates[n_pc]

            # Plot the result and save it as a PDF and as a PNG
            for ext in ('pdf', 'png'):
                file_path = (
                    plots_dir / f'pca__{stacking_factor}__{planet_name}.{ext}'
                )
                fig = plot_frame(
                    frame=frame,
                    file_path=file_path,
                    positions=positions,
                    aperture_radius=aperture_radius,
                    snrs=snrs,
                )
                plt.close(fig=fig)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
