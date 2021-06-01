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
from hsr4hci.hdf import save_dict_to_hdf, create_hdf_dir
from hsr4hci.masking import get_roi_mask
from hsr4hci.training import train_all_models
from hsr4hci.units import InstrumentUnitsContext


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
        help='Index of the split to process; must be in [0, n_roi_splits).',
    )
    parser.add_argument(
        '--n-roi-splits',
        type=int,
        default=1,
        help=(
            'Number of splits into which the ROI is divided (e.g., for '
            'parallel training).'
        ),
    )
    parser.add_argument(
        '--hdf-location',
        type=str,
        choices=['local', 'work'],
        default='work',
        help='Where to create the HDF directory: locally or on /work.',
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
    stack, parang, psf_template, observing_conditions, metadata = load_dataset(
        **config['dataset']
    )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Prepare directory for saving HDF files (*before* training)
    # -------------------------------------------------------------------------

    # Create HDF directory (either locally or on /work)
    create_on_work = args.hdf_location == 'work'
    hdf_dir = create_hdf_dir(experiment_dir, create_on_work=create_on_work)

    # Create a directory for the partial result files (on /work)
    partial_dir = hdf_dir / 'partial'
    partial_dir.mkdir(exist_ok=True, parents=True)

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
    # Loop over pixels in ROI and learn half-sibling regression models
    # -------------------------------------------------------------------------

    # Set up a BaseModelCreator to create instances of our base model
    base_model_creator = BaseModelCreator(**config['base_model'])

    # Train models and get residuals for them
    print('\nTraining models:', flush=True)
    with instrument_units_context:
        partial_residuals = train_all_models(
            roi_mask=roi_mask,
            stack=stack,
            parang=parang,
            psf_template=psf_template,
            obscon_array=observing_conditions.as_array(selected_keys),
            selection_mask_config=config['selection_mask'],
            base_model_creator=base_model_creator,
            n_train_splits=config['n_train_splits'],
            train_mode=config['train_mode'],
            n_signal_times=config['n_signal_times'],
            n_roi_splits=n_roi_splits,
            roi_split=roi_split,
            return_format='partial',
        )
    print()

    # -------------------------------------------------------------------------
    # Save residuals to an HDF file
    # -------------------------------------------------------------------------

    # Finally, save residuals to HDF (in the partial directory)
    print('Saving residuals...', end=' ', flush=True)
    file_name = f'residuals_{roi_split + 1:04d}-{n_roi_splits:04d}.hdf'
    file_path = partial_dir / file_name
    save_dict_to_hdf(dictionary=partial_residuals, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
