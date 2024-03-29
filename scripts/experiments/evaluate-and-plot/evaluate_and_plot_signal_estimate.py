"""
Evaluate (= compute logFPF) and plot signal estimate.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Any

import argparse
import json
import os
import time

from astropy.units import Quantity
from scipy.spatial.distance import euclidean

import numpy as np

from hsr4hci.config import load_config
from hsr4hci.data import load_psf_template, load_metadata, load_planets
from hsr4hci.metrics import compute_metrics
from hsr4hci.fits import read_fits
from hsr4hci.masking import get_roi_mask
from hsr4hci.plotting import plot_frame
from hsr4hci.psf import get_psf_fwhm
from hsr4hci.units import InstrumentUnitsContext


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nEVALUATE AND PLOT SIGNAL ESTIMATE\n', flush=True)

    # -------------------------------------------------------------------------
    # Set up parser to get command line arguments
    # -------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-dir',
        type=str,
        required=True,
        metavar='PATH',
        help='(Absolute) path to experiment directory.',
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load experiment configuration and data
    # -------------------------------------------------------------------------

    # Get experiment directory
    experiment_dir = Path(os.path.expanduser(args.experiment_dir))
    if not experiment_dir.exists():
        raise NotADirectoryError(f'{experiment_dir} does not exist!')

    # Get path to results directory
    results_dir = experiment_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    config = load_config(experiment_dir / 'config.json')
    print('Done!', flush=True)

    # Load the PSF template and estimate its FWHM
    print('Loading PSF template...', end=' ', flush=True)
    psf_template = load_psf_template(**config['dataset']).squeeze()
    psf_fwhm = round(get_psf_fwhm(psf_template), 2)
    print(f'Done! (psf_radius = {psf_fwhm})', flush=True)

    # Load metadata and set up the unit conversion context for this data set
    print('Loading metadata and setting up units...', end=' ', flush=True)
    metadata = load_metadata(**config['dataset'])
    instrument_unit_context = InstrumentUnitsContext(
        pixscale=Quantity(metadata['PIXSCALE'], 'arcsec / pix'),
        lambda_over_d=Quantity(metadata['LAMBDA_OVER_D'], 'arcsec'),
    )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Load signal estimate from FITS and compute SNRs
    # -------------------------------------------------------------------------

    # Load signal estimate
    print('Loading signal estimate...', end=' ', flush=True)
    file_path = results_dir / 'signal_estimate.fits'
    signal_estimate = read_fits(file_path=file_path, return_header=False)
    frame_size = (signal_estimate.shape[0], signal_estimate.shape[1])
    print('Done!', flush=True)

    # Construct the mask for the region of interest (ROI)
    with instrument_unit_context:
        roi_mask = get_roi_mask(
            mask_size=frame_size,
            inner_radius=Quantity(*config['roi_mask']['inner_radius']),
            outer_radius=Quantity(*config['roi_mask']['outer_radius']),
        )

    # Load information about the planets in the dataset
    planets = load_planets(**config['dataset'])

    # Get information about label positions from config file
    try:
        label_positions = config['plotting']['label_positions']
    except KeyError:
        label_positions = None

    # Store labels and positions for plot
    labels = []
    positions = []

    # Loop over all planets in the data set
    print('Computing metrics...', end=' ', flush=True)
    for name, parameters in planets.items():

        # Compute the expected planet position in Cartesian coordinates
        planet_position = (
            Quantity(parameters['separation'], 'arcsec'),
            Quantity(parameters['position_angle'], 'degree'),
        )

        # Compute the metrics
        with instrument_unit_context:
            metrics, position_dict = compute_metrics(
                frame=signal_estimate,
                polar_position=planet_position,
                aperture_radius=Quantity(psf_fwhm / 2, 'pixel'),
                search_radius=Quantity(2, 'pixel'),
                exclusion_angle=None,
            )

        # Create label and store it
        log_fpf = metrics["log_fpf"]
        label = (
            rf'${log_fpf["median"]:.1f}'
            rf'^{{+{log_fpf["max"] - log_fpf["median"]:.1f}}}'
            rf'_{{-{log_fpf["median"] - log_fpf["min"]:.1f}}}$'
        )
        labels.append(label)
        positions.append(position_dict['final']['cartesian'])

        # Save metrics to JSON in results directory
        file_path = results_dir / f'metrics__{name}.json'
        with open(file_path, 'w') as json_file:
            json.dump(metrics, json_file, indent=2)

        # Create a new dictionary which contains the values from the
        # positions_dict but converted to float so that we can save it
        # as a JSON file.
        positions_json: Any = dict(initial={}, final={}, distance=np.nan)
        with instrument_unit_context:

            # Make the positions dictionary serializable (i.e., convert the
            # astropy Quantity objects to regular floats in the right units
            positions_json['final']['polar'] = [
                position_dict['final']['polar'][0].to('arcsec').value,
                position_dict['final']['polar'][1].to('degree').value,
            ]
            positions_json['initial']['polar'] = [
                position_dict['initial']['polar'][0].to('arcsec').value,
                position_dict['initial']['polar'][1].to('degree').value,
            ]

            # Ensure that the polar angle is positive
            if positions_json['final']['polar'][1] < 0:
                positions_json['final']['polar'][1] += 360

            # Compute the distance (in pixels) between the initial and the
            # final position, and add it to the positions_dict
            positions_json['distance'] = euclidean(
                position_dict['initial']['cartesian'],
                position_dict['final']['cartesian']
            )

        # Save positions to JSON in results directory
        file_path = results_dir / f'positions__{name}.json'
        with open(file_path, 'w') as json_file:
            json.dump(positions_json, json_file, indent=2)

    print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Create plot of signal estimate
    # -------------------------------------------------------------------------

    # Ensure that there exists a plots directory in the experiment directory
    plots_dir = experiment_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Apply ROI mask to signal estimate (mostly for PCA plots)
    signal_estimate[~roi_mask] = np.nan

    # Create plot (including SNR etc.)
    print('Creating plot of signal estimate...', end=' ', flush=True)
    file_path = plots_dir / 'signal_estimate.pdf'
    plot_frame(
        frame=signal_estimate,
        file_path=file_path,
        aperture_radius=psf_fwhm,
        pixscale=metadata['PIXSCALE'],
        positions=positions,
        labels=labels,
        label_positions=label_positions,
        add_colorbar=True,
        use_logscale=False,
    )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
