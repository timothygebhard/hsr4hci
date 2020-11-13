"""
Train a collection of half-sibling regression models.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from bisect import insort_left
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from shutil import copyfile
from typing import Dict, List, Tuple, Union

import argparse
import os
import time

from astropy.units import Quantity
from tqdm import tqdm

import numpy as np

from hsr4hci.utils.config import load_config
from hsr4hci.utils.data import load_data
from hsr4hci.utils.hdf import save_dict_to_hdf
from hsr4hci.utils.importing import get_member_by_name
from hsr4hci.utils.masking import (
    get_positions_from_mask,
    get_roi_mask,
)
from hsr4hci.utils.observing_conditions import dict2array
from hsr4hci.utils.psf import (
    crop_psf_template,
    get_artificial_psf,
    get_psf_diameter,
)
from hsr4hci.utils.training import (
    get_default_results,
    get_signal_masking_results,
)
from hsr4hci.utils.typehinting import RegressorModel
from hsr4hci.utils.units import set_units_for_instrument


# -----------------------------------------------------------------------------
# FUNCTIONS DEFINITIONS
# -----------------------------------------------------------------------------

def get_arguments() -> argparse.Namespace:
    """
    Parse command line arguments that are passed to the script.

    Returns:
        The command line arguments as a Namespace object.
    """

    # Set up parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        '--split',
        type=int,
        default=0,
        help='Index of the split for which to run the training script.',
    )
    parser.add_argument(
        '--n-splits',
        type=int,
        default=1,
        help=(
            'Number of splits into which the training data is divided to '
            'parallelize the training.'
        ),
    )

    return parser.parse_args()


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
    stack, parang, psf_template, observing_conditions, metadata = load_data(
        **config['dataset']
    )
    print('Done!', flush=True)

    # Load experiment config from JSON
    print('Parsing command line arguments...', end=' ', flush=True)
    args = get_arguments()
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Define various useful shortcuts
    # -------------------------------------------------------------------------

    # Quantities related to the size of the data set
    n_frames, x_size, y_size = stack.shape
    frame_size = (x_size, y_size)
    center = (x_size / 2, y_size / 2)

    # Metadata of the data set
    pixscale = float(metadata['PIXSCALE'])
    lambda_over_d = float(metadata['LAMBDA_OVER_D'])
    field_rotation = np.abs(parang[-1] - parang[0])

    # Parameters from the experiment configuration
    n_signal_times = config['signal_masking']['n_signal_times']
    max_signal_length = config['signal_masking']['max_signal_length']
    n_splits__baseline = config['n_splits']['baseline']
    n_splits__signal_masking = config['n_splits']['signal_masking']

    # -------------------------------------------------------------------------
    # Activate the unit conversions for this instrument
    # -------------------------------------------------------------------------

    set_units_for_instrument(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
        verbose=False,
    )

    # -------------------------------------------------------------------------
    # Prepare observing conditions for use as predictors
    # -------------------------------------------------------------------------

    # Select observing conditions and construct a 2D array from them
    print('Selecting observing conditions...', end=' ', flush=True)
    obscon_array = dict2array(
        observing_conditions=observing_conditions,
        n_frames=stack.shape[0],
        selected_keys=config['observing_conditions']['selected_keys'],
    )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Prepare PSF
    # -------------------------------------------------------------------------

    # In case we do not have a real PSF template, create a fake one
    if psf_template.shape == (0, 0):

        print('Creating artificial PSF template...', end=' ', flush=True)
        psf_template = get_artificial_psf(
            pixscale=Quantity(pixscale, 'arcsec / pixel'),
            lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
        )
        print('Done!', flush=True)

    # Fit the PSF template with a 2D Moffat function to estimate its diameter
    print('Fitting PSF diameter...', end=' ', flush=True)
    psf_diameter = get_psf_diameter(
        psf_template=psf_template,
        pixscale=pixscale,
        lambda_over_d=lambda_over_d,
    )
    print('Done!', flush=True)

    # Crop the PSf template
    print('Cropping PSF template...', end=' ', flush=True)
    psf_cropped = crop_psf_template(
        psf_template=psf_template, psf_radius=Quantity(1.0, 'lambda_over_d'),
    )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Set up a function that returns a new instance of our base model
    # -------------------------------------------------------------------------

    @lru_cache(maxsize=1)
    def get_model_instance() -> RegressorModel:
        """
        Get a new instance of the base model defined in the config.

        Returns:
            An instance of a regression method (e.g., from sklearn) that
            must provide the .fit() and .predict() methods.
        """

        # Get the model class and the model parameters
        module_name = config['base_model']['module']
        class_name = config['base_model']['class']
        model_class = get_member_by_name(
            module_name=module_name, member_name=class_name
        )
        model_parameters = deepcopy(config['base_model']['parameters'])

        # Augment the model parameters:
        # For RidgeCV models, we have to parse the `alphas` parameter (i.e.,
        # the regularization strengths) into a geometrically spaced array
        if class_name == 'RidgeCV':
            model_parameters['alphas'] = np.geomspace(
                *model_parameters['alphas']
            )

        # Instantiate a new model of the given class with the desired params
        model: RegressorModel = model_class(**model_parameters)

        return model

    # -------------------------------------------------------------------------
    # Set up the mask for the region of interest (ROI)
    # -------------------------------------------------------------------------

    # Define a mask for the ROI
    roi_mask = get_roi_mask(
        mask_size=frame_size,
        inner_radius=Quantity(*config['roi_mask']['inner_radius']),
        outer_radius=Quantity(*config['roi_mask']['outer_radius']),
    )

    # Get a list of the positions in the ROI
    roi_positions = get_positions_from_mask(mask=roi_mask)

    # -------------------------------------------------------------------------
    # Ensure results directory exists, and copy over configuration file
    # -------------------------------------------------------------------------

    # Create backup copy of the config file
    print('Backing up experiment configuration...', end=' ', flush=True)
    copyfile(experiment_dir / 'config.json', results_dir / 'config.json')
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Print information about the data set and the experiment configuration
    # -------------------------------------------------------------------------

    # Define shortcuts for printing
    target_star = metadata['TARGET_STAR']
    filter_name = metadata['FILTER']
    stacking_factor = config['dataset']['stacking_factor']
    obscon_keys = sorted(list(observing_conditions.keys()))
    obscon_keys_str = '\n'.join([f'  {_}' for _ in obscon_keys])
    model_config = config['base_model']
    model_name = model_config['module'] + '.' + model_config['class']
    model_params = config['base_model']['parameters']

    # Print information about data set and experiment configuration
    print('')
    print(80 * '-')
    print('Parallelization:')
    print('Data set:')
    print(f'  Target:                   {target_star} ({filter_name})')
    print(f'  Date:                     {metadata["DATE"]}')
    print(f'  PIXSCALE:                 {pixscale} arcsec / pixel')
    print(f'  LAMBDA_OVER_D:            {lambda_over_d} arcsec')
    print('Stack:')
    print(f'  Shape:                    {stack.shape}')
    print(f'  Stacking factor:          {stacking_factor}')
    print('Observing conditions:')
    print(f'{obscon_keys_str}')
    print('Model:')
    print(f'  Class:                    {model_name}')
    print(f'  Parameters:               {model_params})')
    print('Signal masking:')
    print(f'  n_signal_times:           {n_signal_times}')
    print(f'  max_signal_length:        {max_signal_length}')
    print(f'  n_splits__baseline:       {n_splits__baseline}')
    print(f'  n_splits__signal_masking: {n_splits__signal_masking}')
    print(80 * '-')
    print('')

    # -------------------------------------------------------------------------
    # Loop over pixels in ROI and learn half-sibling regression models
    # -------------------------------------------------------------------------

    # Initialize dictionary that will hold the results. We initialize it with
    # an array holding the signal times for which we have computed model with
    # signal masking, as well as a mask in which we keep track of the pixels
    # that we have processed with this script. This is useful if we are running
    # multiple instances of the training script in parallel, because it allows
    # us to save the data more space-efficient.
    ResultsType = Dict[
        str, Union[np.ndarray, Dict[str, Union[list, np.ndarray]]]
    ]
    results: ResultsType = dict(
        stack_shape=np.array(stack.shape),
        signal_times=np.linspace(0, n_frames - 1, n_signal_times),
    )

    # Create an auxiliary array which tells us the index that each spatial
    # position will end up at when we reshape a stack-shaped 3D array to a 2D
    # array by flattening the spatial dimensions.
    lookup_column_indices = np.arange(x_size * y_size).astype(int)
    lookup_column_indices = lookup_column_indices.reshape(frame_size)

    # Keep track of the indices of the columns we are processing
    processed_column_indices: List[int] = list()

    # Loop over positions in the ROI and process each pixel individually
    print(f'Training models (split {args.split + 1} of {args.n_splits}):')
    for _position in tqdm(
        roi_positions[args.split::args.n_splits], ncols=80
    ):

        # ---------------------------------------------------------------------
        # Preliminaries: Define shortcuts, get indices
        # ---------------------------------------------------------------------

        # Typecast the dummy position to make sure mypy understands the type
        # of `position` after coming out of the tqdm iterator
        position: Tuple[int, int] = (int(_position[0]), int(_position[1]))

        # Get the column index of the current position
        column_idx = lookup_column_indices[position[0], position[1]]

        # Get the index at which we have to insert the results for the current
        # position in the lists in which we keep track of the results
        insert_idx = np.searchsorted(processed_column_indices, column_idx)

        # Add this index to the list of indices that we have processed and
        # make sure that the list stays sorted (insort_left() works in-place!)
        insort_left(processed_column_indices, insert_idx)

        # ---------------------------------------------------------------------
        # Compute and store the default results
        # ---------------------------------------------------------------------

        # Get default results
        default_results = get_default_results(
            position=position,
            stack=stack,
            parang=parang,
            obscon_array=obscon_array,
            selection_mask_config=config['selection_mask'],
            get_model_instance=get_model_instance,
            n_splits=n_splits__baseline,
            psf_diameter=psf_diameter,
        )

        # If 'baseline' does not yet exist in the results dictionary (i.e.,
        # during the first iteration of the loop), initialize with empty lists
        if 'default' not in results.keys():
            results['default'] = dict(
                predictions=[],
                residuals=[],
                mask=np.full(frame_size, False),
            )

        # Now insert the results for this (spatial) position at the correct
        # position in the results list; "correct" meaning that if we turn this
        # list into a 2D numpy array and use the `results['mask']` to assign
        # it to a subset of a stack-shaped 3D array, everything ends up at the
        # expected position.
        for _ in ('predictions', 'residuals'):
            results['default'][_].insert(insert_idx, default_results[_])
        np.asarray(results['default']['mask'])[position[0], position[1]] = 1

        # ---------------------------------------------------------------------
        # Compute and store the results based on masking a potential signal
        # ---------------------------------------------------------------------

        # Get results based on signal masking
        signal_masking_results = get_signal_masking_results(
            position=position,
            stack=stack,
            parang=parang,
            obscon_array=obscon_array,
            selection_mask_config=config['selection_mask'],
            get_model_instance=get_model_instance,
            n_signal_times=n_signal_times,
            frame_size=frame_size,
            psf_diameter=psf_diameter,
            psf_cropped=psf_cropped,
            n_splits=n_splits__signal_masking,
            max_signal_length=max_signal_length,
        )

        # Store result based on signal masking
        for key in signal_masking_results.keys():

            if key not in results.keys():
                results[key] = dict(
                    signal_time=[],
                    signal_mask=[],
                    predictions=[],
                    residuals=[],
                    mask=np.full(frame_size, False),
                )

            for _ in (
                'signal_time',
                'signal_mask',
                'predictions',
                'residuals',
            ):
                results[key][_].insert(
                    insert_idx, signal_masking_results[key][_]
                )
            np.asarray(results[key]['mask'])[position[0], position[1]] = 1

    print()

    # -------------------------------------------------------------------------
    # Loop over the whole results dictionary and convert lists to arrays
    # -------------------------------------------------------------------------

    print('Converting results to numpy arrays...', end=' ', flush=True)
    for key, value in results.items():

        # We only need to process the second-level dictionary; the first-level
        # entries (stack_shape, signal_times and mask) are numpy arrays already
        if isinstance(value, dict):

            # Convert all lists in second-level dictionaries to 2D numpy
            # arrays. The transpose is necessary when reconstructing the 3D
            # array from the 2D arrays using the position mask.
            for name, item in value.items():
                if name == 'mask':
                    continue
                results[key][name] = np.array(item).T

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Save results to an HDF file
    # -------------------------------------------------------------------------

    print('Saving results...', end=' ', flush=True)
    hdf_dir = results_dir / 'hdf'
    hdf_dir.mkdir(exist_ok=True)
    file_path = hdf_dir / f'results_{args.split+1:03d}-{args.n_splits:03d}.hdf'
    save_dict_to_hdf(dictionary=results, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
