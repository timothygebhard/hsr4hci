"""
Loop over all sub-experiments for an algorithm and regenerate the
selection mask and the final signal estimate. This can make sense if
the procedure to determine the selection mask (but not the training
procedure) has changed.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import os
import time

from astropy.units import Quantity

from joblib import Parallel, delayed
from tqdm.auto import tqdm

import numpy as np

from hsr4hci.config import load_config
from hsr4hci.data import load_psf_template, load_parang, load_metadata
from hsr4hci.derotating import derotate_combine
from hsr4hci.fits import read_fits, save_fits
from hsr4hci.masking import get_roi_mask
from hsr4hci.residuals import get_residual_selection_mask
from hsr4hci.units import InstrumentUnitsContext


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nRECREATE SELECTION MASKS AND SIGNAL ESTIMATES\n')

    # -------------------------------------------------------------------------
    # Parse command line arguments
    # -------------------------------------------------------------------------

    # Set up a parser and parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        required=True,
        help='Main directory of the experiment set.',
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=32,
        help='Number of parallel jobs for processing the results.',
    )
    args = parser.parse_args()

    # Define shortcuts
    directory = Path(args.directory).resolve()
    n_jobs = args.n_jobs

    # Make sure the main directory (where the base_config.json resides) exists
    if not directory.exists():
        raise RuntimeError(f'{directory} does not exist!')

    # -------------------------------------------------------------------------
    # Load the base_config.json and determine the ROI mask
    # -------------------------------------------------------------------------

    # Load the base_config to get the data set properties
    base_config = load_config(directory / 'base_config.json')

    # Load the parallactic angles and the PSF template
    parang = load_parang(**base_config['dataset'])
    psf_template = load_psf_template(**base_config['dataset'])
    metadata = load_metadata(**base_config['dataset'])

    # Define the unit conversion context for this data set
    instrument_units_context = InstrumentUnitsContext(
        pixscale=Quantity(metadata['PIXSCALE'], 'arcsec / pixel'),
        lambda_over_d=Quantity(metadata['LAMBDA_OVER_D'], 'arcsec'),
    )

    # Construct the mask for the region of interest (ROI)
    with instrument_units_context:
        roi_mask = get_roi_mask(
            mask_size=base_config['dataset']['frame_size'],
            inner_radius=Quantity(*base_config['roi_mask']['inner_radius']),
            outer_radius=Quantity(*base_config['roi_mask']['outer_radius']),
        )

    # -------------------------------------------------------------------------
    # Find experiment folders for which we have results
    # -------------------------------------------------------------------------

    # Define directory that holds all the experiment folders
    experiments_dir = directory / 'experiments'

    # Get a list of all experiment folders (except no_fake_planets)
    experiment_dirs = list(
        filter(
            lambda _: os.path.isdir(_),
            sorted([experiments_dir / _ for _ in os.listdir(experiments_dir)]),
        )
    )

    # -------------------------------------------------------------------------
    # Loop over experiment folders, get signal estimate and compute throughput
    # -------------------------------------------------------------------------

    def process_directory(experiment_dir: Path) -> None:
        """
        Auxiliary function to process an experiment directory: load MF,
        compute selection mask, assemble residuals, and compute signal
        estimate.
        """

        # Define path to results directory
        results_dir = experiment_dir / 'results'

        # Load match fraction
        file_path = results_dir / 'median_mf.fits'
        match_fraction = read_fits(file_path=file_path, return_header=False)

        # Compute residual selection mask
        selection_mask, _, _, _, _ = get_residual_selection_mask(
            match_fraction=match_fraction,
            parang=parang,
            psf_template=psf_template,
        )

        # Save the residual selection mask as a FITS file
        file_path = results_dir / 'selection_mask.fits'
        save_fits(array=selection_mask.astype(int), file_path=file_path)

        # Load the residual stacks (default and hypothesis-based)
        file_path = results_dir / 'default_residuals.fits'
        default_residuals = read_fits(
            file_path=file_path, return_header=False
        )
        file_path = results_dir / 'hypothesis_residuals.fits'
        hypothesis_residuals = read_fits(
            file_path=file_path, return_header=False
        )

        # Construct the final residual stack based on the selection mask
        residuals = np.copy(default_residuals)
        residuals[:, selection_mask] = hypothesis_residuals[:, selection_mask]

        # Merge the residuals into a signal estimate
        signal_estimate = derotate_combine(
            stack=residuals, parang=parang, mask=~roi_mask
        )

        # Save the signal_estimate as a FITS file
        file_path = results_dir / 'signal_estimate.fits'
        save_fits(array=signal_estimate, file_path=file_path)

    # Use joblib to process all the experiments in parallel
    print('Processing directories in parallel:')
    Parallel(n_jobs=n_jobs)(
        delayed(process_directory)(_) for _ in tqdm(experiment_dirs, ncols=80)
    )

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
