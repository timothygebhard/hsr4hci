"""
Run stage 2 of the pipeline (find hypotheses, compute match fractions).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict

import argparse
import os
import time

from astropy.units import Quantity

import numpy as np

from hsr4hci.config import load_config
from hsr4hci.match_fraction import get_all_match_fractions
from hsr4hci.data import load_metadata, load_parang, load_psf_template
from hsr4hci.fits import save_fits
from hsr4hci.hdf import load_dict_from_hdf
from hsr4hci.hypotheses import get_all_hypotheses
from hsr4hci.masking import get_roi_mask
from hsr4hci.units import InstrumentUnitsContext


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nRUN STAGE 2: FIND HYPOTHESES AND COMPUTE MFs\n', flush=True)

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
    parser.add_argument(
        '--roi-split',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--n-roi-splits',
        type=int,
        default=1,
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load experiment configuration and data
    # -------------------------------------------------------------------------

    # Get experiment directory
    experiment_dir = Path(os.path.expanduser(args.experiment_dir))
    if not experiment_dir.exists():
        raise NotADirectoryError(f'{experiment_dir} does not exist!')

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    config = load_config(experiment_dir / 'config.json')
    print('Done!', flush=True)

    # Load frames, parallactic angles, etc. from HDF file
    print('Loading data set...', end=' ', flush=True)
    parang = load_parang(**config['dataset'])
    metadata = load_metadata(**config['dataset'])
    psf_template = load_psf_template(**config['dataset'])
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Define various useful shortcuts; activate unit conversions
    # -------------------------------------------------------------------------

    # Quantities related to the size of the data set
    frame_size = (
        int(config['dataset']['frame_size'][0]),
        int(config['dataset']['frame_size'][1]),
    )

    # Metadata of the data set
    pixscale = float(metadata['PIXSCALE'])
    lambda_over_d = float(metadata['LAMBDA_OVER_D'])

    # Other shortcuts
    n_signal_times = config['n_signal_times']
    roi_split = args.roi_split
    n_roi_splits = args.n_roi_splits

    # Define the unit conversion context for this data set
    instrument_units_context = InstrumentUnitsContext(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
    )

    # Construct the mask for the region of interest (ROI)
    with instrument_units_context:
        roi_mask = get_roi_mask(
            mask_size=frame_size,
            inner_radius=Quantity(*config['roi_mask']['inner_radius']),
            outer_radius=Quantity(*config['roi_mask']['outer_radius']),
        )

    # -------------------------------------------------------------------------
    # STEP 1: Find hypotheses
    # -------------------------------------------------------------------------

    # Load results. This is very memory-intensive, but read-on-demand seems
    # orders of magnitude slower and thus infeasible
    print('\nLoading results from HDF...', end=' ', flush=True)
    file_path = experiment_dir / 'hdf' / 'residuals.hdf'
    residuals: Dict[str, np.ndarray] = load_dict_from_hdf(file_path)
    print('Done!', flush=True)

    # Find best hypothesis (for specified subset of ROI)
    print('\nFinding best hypothesis for each spatial pixel:', flush=True)
    hypotheses, similarities = get_all_hypotheses(
        roi_mask=roi_mask,
        residuals=residuals,
        parang=parang,
        n_signal_times=n_signal_times,
        frame_size=frame_size,
        psf_template=psf_template,
        n_roi_splits=n_roi_splits,
        roi_split=roi_split,
    )

    # Create (partial) directory for hypotheses
    hypotheses_dir = experiment_dir / 'hypotheses' / 'partial'
    hypotheses_dir.mkdir(exist_ok=True, parents=True)

    # Save hypotheses as a FITS file
    print('\nSaving hypotheses to FITS...', end=' ', flush=True)
    file_name = f'hypotheses_{roi_split + 1:04d}-{n_roi_splits:04d}.fits'
    file_path = hypotheses_dir / file_name
    save_fits(array=hypotheses, file_path=file_path)
    print('Done!', flush=True)

    # Save cosine similarities (of hypotheses) as a FITS file
    print('Saving similarities to FITS...', end=' ', flush=True)
    file_name = f'similarities_{roi_split + 1:04d}-{n_roi_splits:04d}.fits'
    file_path = hypotheses_dir / file_name
    save_fits(array=similarities, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # STEP 3: Compute match fractions and save them to FITS
    # -------------------------------------------------------------------------

    # Compute match fraction (for specified subset of ROI)
    print('\nComputing match fractions:', flush=True)
    mean_mf, median_mf, _ = get_all_match_fractions(
        residuals=residuals,
        roi_mask=roi_mask,
        hypotheses=hypotheses,
        parang=parang,
        psf_template=psf_template,
        frame_size=frame_size,
        n_roi_splits=n_roi_splits,
        roi_split=roi_split,
    )

    # Create (partial) matches directory
    partial_dir = experiment_dir / 'match_fractions' / 'partial'
    partial_dir.mkdir(exist_ok=True, parents=True)

    # Save match fraction(s) as FITS file
    print('Saving mean match fractions to FITS...', end=' ', flush=True)
    file_name = f'mean_mf_{roi_split + 1:04d}-{n_roi_splits:04d}.fits'
    file_path = partial_dir / file_name
    save_fits(array=mean_mf, file_path=file_path)
    print('Done!', flush=True)

    print('Saving median match fractions to FITS...', end=' ', flush=True)
    file_name = f'median_mf_{roi_split + 1:04d}-{n_roi_splits:04d}.fits'
    file_path = partial_dir / file_name
    save_fits(array=median_mf, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
