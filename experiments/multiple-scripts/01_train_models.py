"""
Train a collection of half-sibling regression models.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import os
import time

from astropy.units import Quantity

from hsr4hci.base_models import BaseModelCreator
from hsr4hci.config import load_config
from hsr4hci.data import load_dataset
from hsr4hci.hdf import save_dict_to_hdf
from hsr4hci.masking import get_roi_mask
from hsr4hci.training import train_all_models
from hsr4hci.units import set_units_for_instrument


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nTRAIN HALF-SIBLING REGRESSION MODELS\n', flush=True)

    # -------------------------------------------------------------------------
    # Load experiment configuration and data; parse command line arguments
    # -------------------------------------------------------------------------

    # Define paths for experiment folder and results folder
    experiment_dir = Path(os.path.realpath(__file__)).parent
    results_dir = experiment_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    config = load_config(experiment_dir / 'config.json')
    print('Done!', flush=True)

    # Load frames, parallactic angles, etc. from HDF file
    print('Loading data set...', end=' ', flush=True)
    stack, parang, psf_template, observing_conditions, metadata = load_dataset(
        **config['dataset']
    )
    print('Done!', flush=True)

    # Get command line arguments
    print('Parsing command line arguments...', end=' ', flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi-split', type=int, default=0)
    parser.add_argument('--n-roi-splits', type=int, default=1)
    args = parser.parse_args()
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Define various useful shortcuts; activate unit conversions
    # -------------------------------------------------------------------------

    # Quantities related to the size of the data set
    n_frames, x_size, y_size = stack.shape
    frame_size = (x_size, y_size)

    # Metadata of the data set
    pixscale = float(metadata['PIXSCALE'])
    lambda_over_d = float(metadata['LAMBDA_OVER_D'])

    # Other shortcuts
    selected_keys = config['observing_conditions']['selected_keys']
    roi_split = args.roi_split
    n_roi_splits = args.n_roi_splits

    # Activate the unit conversions for this instrument
    set_units_for_instrument(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
        verbose=False,
    )

    # -------------------------------------------------------------------------
    # Loop over pixels in ROI and learn half-sibling regression models
    # -------------------------------------------------------------------------

    # Set up a BaseModelCreator to create instances of our base model
    base_model_creator = BaseModelCreator(**config['base_model'])

    # Construct the mask for the region of interest (ROI)
    roi_mask = get_roi_mask(
        mask_size=frame_size,
        inner_radius=Quantity(*config['roi_mask']['inner_radius']),
        outer_radius=Quantity(*config['roi_mask']['outer_radius']),
    )

    print('\nTraining models:', flush=True)
    results = train_all_models(
        roi_mask=roi_mask,
        stack=stack,
        parang=parang,
        psf_template=psf_template,
        obscon_array=observing_conditions.as_array(selected_keys),
        selection_mask_config=config['selection_mask'],
        base_model_creator=base_model_creator,
        n_splits=config['n_splits'],
        mode=config['mode'],
        n_signal_times=config['n_signal_times'],
        n_roi_splits=n_roi_splits,
        roi_split=roi_split,
        return_format='partial',
    )
    print()

    # -------------------------------------------------------------------------
    # Save results to an HDF file
    # -------------------------------------------------------------------------

    # Make sure that the HDF directory (for partial result files) exists
    hdf_dir = results_dir / 'hdf'
    hdf_dir.mkdir(exist_ok=True)

    # Save results to HDF
    print('Saving results...', end=' ', flush=True)
    file_path = hdf_dir / f'results_{roi_split + 1:03d}-{n_roi_splits:03d}.hdf'
    save_dict_to_hdf(dictionary=results, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
