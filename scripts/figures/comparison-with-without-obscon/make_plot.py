"""
Create plot to compare the signal estimate with and without observing
conditions as predictors.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from argparse import ArgumentParser

import time

from astropy.units import Quantity

import matplotlib.pyplot as plt
import numpy as np

from hsr4hci.config import get_hsr4hci_dir
from hsr4hci.data import load_metadata
from hsr4hci.fits import read_fits
from hsr4hci.coordinates import get_center
from hsr4hci.general import crop_center
from hsr4hci.plotting import plot_frame
from hsr4hci.units import InstrumentUnitsContext

import hsr4hci


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
    parser.add_argument('--dataset', type=str, default='beta_pictoris__mp')
    parser.add_argument('--algorithm', type=str, default='signal_masking')
    parser.add_argument('--limit', type=float, default=15)

    # Define shortcuts
    args = parser.parse_args()
    dataset = args.dataset
    algorithm = args.algorithm
    limit = args.limit

    # -------------------------------------------------------------------------
    # Load metadata and compute crop radius in pixels
    # -------------------------------------------------------------------------

    # Load the metadata (to get the pixscale)
    metadata = load_metadata(name_or_path=dataset)

    for i, (name, directory) in enumerate(
        [
            ('without', Path('01_first-results', algorithm, dataset)),
            ('with', Path('03_observing-conditions', algorithm, dataset)),
        ]
    ):

        print(f'Creating plot {name} obscon. ...', end=' ', flush=True)

        # Load and crop the signal estimate
        file_path = (
            get_hsr4hci_dir()
            / 'experiments'
            / directory
            / 'results'
            / 'signal_estimate.fits'
        )
        signal_estimate = read_fits(file_path, return_header=False)
        signal_estimate = crop_center(signal_estimate, (21, 21))

        plot_dir = Path('.') / 'plots' / dataset / algorithm
        plot_dir.mkdir(exist_ok=True, parents=True)

        file_path = plot_dir / f'{name}.pdf'
        fig, ax, cbar = plot_frame(
            frame=signal_estimate,
            positions=[],
            labels=[],
            pixscale=metadata['PIXSCALE'],
            figsize=(4.3 / 2.54, 5.0 / 2.54),
            subplots_adjust=dict(
                left=0.01, top=1.01, right=0.99, bottom=0.07
            ),
            aperture_radius=0,
            scalebar_color='black',
            limits=(-limit, limit),
            file_path=file_path,
        )

        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
