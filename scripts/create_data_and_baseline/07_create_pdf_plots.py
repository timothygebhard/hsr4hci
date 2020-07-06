"""
This script creates PDF versions of the signal estimates with the
highest SNR, both for PCA and for median ADI.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from ast import literal_eval
from pathlib import Path
from typing import List, Tuple

import json
import os
import time

from astropy import units
from photutils import CircularAperture

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hsr4hci.utils.argparsing import get_base_directory
from hsr4hci.utils.fits import read_fits
from hsr4hci.utils.plotting import (
    add_colorbar_to_ax,
    get_cmap,
    MatplotlibColor,
)
from hsr4hci.utils.units import set_units_for_instrument, to_pixel


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def make_plot(
    frame: np.ndarray,
    file_path: str,
    positions: List[Tuple[float, float]],
    aperture_radius: float,
    snrs: List[float],
    color: MatplotlibColor = 'darkgreen',
) -> None:

    # Set up a new figure
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.set_constrained_layout_pads(w_pad=0, h_pad=0)

    # Define aperture to get plot limits, and also to for plotting later
    aperture = CircularAperture(positions=positions, r=aperture_radius)
    photometry, _ = aperture.do_photometry(data=frame)

    # Determine the limits for the color map
    limit = 1.2 * np.nanmax(photometry) / aperture.area

    # Create the actual plot and add a colorbar
    img = plt.imshow(
        X=frame,
        origin='lower',
        cmap=get_cmap(),
        vmin=-limit,
        vmax=limit,
        interpolation='none',
    )
    add_colorbar_to_ax(img=img, fig=fig, ax=ax)

    # Plot the optimal signal apertures and the resulting SNR
    aperture.plot(axes=ax, **dict(color=color, lw=1, ls='-'))
    for snr, position in zip(snrs, positions):

        # Compute position of the label containing the SNR
        angle = np.arctan2(
            position[1] - frame.shape[1] / 2, position[0] - frame.shape[0] / 2
        )
        x = position[0] + 3 * aperture_radius * np.cos(angle)
        y = position[1] + 3 * aperture_radius * np.sin(angle)

        # Actually add the label with the SNR at this position
        ax.text(
            x=x,
            y=y,
            s=f'{snr:.1f}',
            ha='center',
            va='center',
            color='white',
            fontsize=6,
            bbox=dict(
                facecolor=color, edgecolor='none', boxstyle='round,pad=0.15',
            ),
        )

        # Draw connection between label and aperture
        ax.plot(
            [position[0] + aperture_radius * np.cos(angle), x],
            [position[1] + aperture_radius * np.sin(angle), y],
            color=color,
            lw=1,
        )

    # Remove ax ticks
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        labelleft=False,
        labelbottom=False,
    )

    # Save the results
    plt.savefig(file_path, bbox_inches='tight', pad=0)

    # Close the figure
    plt.close(fig=fig)


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nRUN MEDIAN ADI AND COMPUTE SIGNAL ESTIMATES\n', flush=True)

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

    # Now, apply unit conversions to astropy.units:
    # First, convert pixscale and lambda_over_d to astropy.units.Quantity. This
    # is a bit cumbersome, because in the meta data, we cannot use the usual
    # convention to specify units, as the meta data are also written to the HDF
    # file. Hence, we must hard-code the unit conventions here.
    config['metadata']['PIXSCALE'] = units.Quantity(
        config['metadata']['PIXSCALE'], 'arcsec / pixel'
    )
    config['metadata']['LAMBDA_OVER_D'] = units.Quantity(
        config['metadata']['LAMBDA_OVER_D'], 'arcsec'
    )

    # Use this to set up the instrument-specific conversion factors. We need
    # this here to that we can parse "lambda_over_d" as a unit in the config.
    set_units_for_instrument(
        pixscale=config['metadata']['PIXSCALE'],
        lambda_over_d=config['metadata']['LAMBDA_OVER_D'],
    )

    # Define shortcuts
    stacking_factors = config['stacking_factors']
    planet_names = list(config['evaluation']['planets'].keys())
    aperture_radius = config['evaluation']['snr_options']['aperture_radius']
    aperture_radius = to_pixel(units.Quantity(*aperture_radius))

    # -------------------------------------------------------------------------
    # Loop over stacking factors and create plots
    # -------------------------------------------------------------------------

    # Prepare results directory
    plots_dir = os.path.join(base_dir, 'plots')
    Path(plots_dir).mkdir(exist_ok=True)

    for stacking_factor in stacking_factors:

        # ---------------------------------------------------------------------
        # Create plots for median ADI results
        # ---------------------------------------------------------------------

        # Define directory that contains results for this stacking factor
        stacking_dir = os.path.join(
            base_dir, 'median_adi_baselines', f'stacked_{stacking_factor}'
        )

        # Collect optimized positions and SNRs for all planets
        positions, snrs = list(), list()
        for planet_name in planet_names:
            file_path = os.path.join(stacking_dir, f'{planet_name}.csv')
            dataframe = pd.read_csv(file_path, header=None).set_index(0).T
            positions.append(literal_eval(dataframe['new_position'].values[0]))
            snrs.append(float(dataframe['snr'].values[0]))

        # Read in the frame containing the signal estimate
        file_path = os.path.join(stacking_dir, 'signal_estimate.fits')
        frame = read_fits(file_path=file_path)

        # Plot the result and save it as a PDF
        file_path = os.path.join(plots_dir, f'madi__{stacking_factor}.pdf')
        make_plot(
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
        stacking_dir = os.path.join(
            base_dir, 'pca_baselines', f'stacked_{stacking_factor}'
        )

        # Read in the CSV file with the figures of merit into a dataframe
        file_path = os.path.join(stacking_dir, 'figures_of_merit.csv')
        dataframe = pd.read_csv(
            filepath_or_buffer=file_path,
            sep='\t',
            header=[0, 1],
            index_col=[0, 1],
        )

        # Read the stack of signal estimates
        file_path = os.path.join(stacking_dir, 'signal_estimates.fits')
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
                    literal_eval(
                        dataframe[planet_name]['new_position'].values[n_pc]
                    )
                )
                snrs.append(dataframe[planet_name]['snr'].values[n_pc])

            # Select the signal estimate that matches the given number of PCs
            frame = signal_estimates[n_pc]

            # Plot the result and save it as a PDF
            file_path = os.path.join(
                plots_dir, f'pca__{stacking_factor}__{planet_name}.pdf'
            )
            make_plot(
                frame=frame,
                file_path=file_path,
                positions=positions,
                aperture_radius=aperture_radius,
                snrs=snrs,
            )

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'This took {time.time() - script_start:.1f} seconds!\n')
