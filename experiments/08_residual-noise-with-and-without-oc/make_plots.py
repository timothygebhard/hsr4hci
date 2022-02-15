"""
Create plots of signal estimate (or rather: the residual noise) with
and without the observing conditions, for different binning factors.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import time

from hsr4hci.data import load_metadata
from hsr4hci.fits import read_fits
from hsr4hci.general import crop_center
from hsr4hci.plotting import plot_frame


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
    # Parse command line arguments; load metadata; prepare plots directory
    # -------------------------------------------------------------------------

    # Set up a parser and parse the command line arguments
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='beta_pictoris__mp')
    parser.add_argument('--limit', type=float, default=15)

    # Define shortcuts
    args = parser.parse_args()
    dataset = args.dataset
    limit = args.limit

    # Load the metadata (to get the pixscale)
    metadata = load_metadata(name_or_path=dataset)

    # Make sure the plots directory exists
    plots_dir = Path('.') / dataset / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Loop over experiments and create plots
    # -------------------------------------------------------------------------

    for oc, bf in product(('with', 'without'), (1, 10, 100, 1000)):

        print(f'Creating plot: {oc} OC, BF: {bf} ...', end=' ', flush=True)

        # Load and crop the signal estimate
        file_path = (
            Path('.')
            / dataset
            / 'experiments'
            / f'binning_factor-{bf}'
            / f'{oc}-oc'
            / 'results'
            / 'signal_estimate.fits'
        )

        # Read in and crop the signal estimate
        signal_estimate = read_fits(file_path, return_header=False)
        signal_estimate = crop_center(signal_estimate, (17, 17))

        # Create a plot and save it as a PDF
        file_path = plots_dir / f'{oc}-oc__binning_factor-{bf}.pdf'
        plot_frame(
            frame=signal_estimate,
            positions=[],
            labels=[],
            pixscale=metadata['PIXSCALE'],
            figsize=(4.3 / 2.54, 4.3 / 2.54),
            subplots_adjust=dict(left=0.01, top=0.99, right=0.99, bottom=0.01),
            aperture_radius=0,
            add_colorbar=False,
            scalebar_color='black',
            limits=(-limit, limit),
            file_path=file_path,
        )

        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
