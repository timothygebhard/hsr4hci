"""
Utility functions training half-sibling regression models.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from bisect import insort_left
from typing import Any, Dict, List, Tuple, Union

from astropy.units import Quantity
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

import numpy as np

from hsr4hci.utils.base_models import BaseModelCreator
from hsr4hci.utils.masking import get_selection_mask, get_positions_from_mask
from hsr4hci.utils.psf import get_psf_radius
from hsr4hci.utils.signal_masking import get_signal_masks
from hsr4hci.utils.splitting import TrainTestSplitter


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def train_models(
    roi_mask: np.ndarray,
    stack: np.ndarray,
    parang: np.ndarray,
    obscon_array: np.ndarray,
    selection_mask_config: Dict[str, Any],
    base_model_creator: BaseModelCreator,
    n_signal_times: int,
    psf_template: np.ndarray,
    n_splits: Dict[str, int],
    max_signal_length: float,
    n_roi_splits: int = 1,
    roi_split: int = 0,
    return_format: str = 'full',
) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
    """
    This function wraps the loop over the ROI and runs the default and
    signal masking-based training procedures for every pixel.

    It also wraps the special provisions that are required when running
    the training process in parallel, meaning it can return the results
    in the special "partial" format which is more space-efficient but
    requires an additional merging step.

    Args:
        roi_mask:
        stack:
        parang:
        obscon_array:
        selection_mask_config:
        base_model_creator:
        n_signal_times:
        psf_template:
        n_splits:
        max_signal_length:
        n_roi_splits: Number of splits for the region of interest (ROI).
            If this number if greater than one, this function does not
            process the entire ROI, but only every `n_roi_splits`-th
            pixel in it. This can be used to parallelize the training.
        roi_split: If the ROI is split into `n_roi_splits` parts for
            training, this parameter controls which of the splits is
            processed by this function (namely, the `roi_split`-th).
            Example: if `n_roi_splits` == 4`, then we need to call this
            function with four times with `roi_split == 0, ..., 3` to
            really process all pixels in the ROI.
        return_format: The format in which the results are returned.
            Must be either "partial" or "full". The difference are
            as follows:
                partial: The returned arrays do not have the same shape
                    as the stack (i.e., `(n_frames, width, height)`),
                    but are 2D arrays that have the shape:
                        `(n_frames, n_pixels_in_split)`
                    where `n_pixels_in_split` is approximately equal to
                    `pixels_in_roi / n_splits`. This format is useful
                    when the training procedure is run in parallel. In
                    this case, the results (from this function) of each
                    node can be stored as its own HDF file (without too
                    much overhead), and then merged into the "proper"
                    results file.
                full:

    Returns:
        A dictionary that contains the training results. Depending on
        the `return_format`, it can take two different forms.

        For `return_format == "partial"`:

        ```
        {
            "stack_shape": A 3-tuple containing the shape of the
                original stack. This is necessary when partial results
                that have been saved as HDF files should be merged
                again later on.
            "signal_times": A numpy array with shape `(n_signal_times,)`
                containing the signal times.
            "default": {
                "predictions": A 2D numpy array with shape `(n_frames,
                    n_pixels_in_split)` that contains the predictions
                    for the "default" case (i.e., no signal masking).
                "residuals": A 2D numpy array with shape `(n_frames,
                    n_pixels_in_split)` that contains the residuals
                    for the "default" case (i.e., no signal masking).
                "mask": A 2D numpy array with shape `(width, height)`
                    that contains a binary mask indicating the subset
                    of the ROI that was processed by this function.
                    This mask can be used to insert the `predictions`
                    and `residuals` at the correct positions  inside a
                    `stack`-shaped array.
            }
            "0": {
                "predictions": A 2D numpy array with shape `(n_frames,
                    n_pixels_in_split)` that contains the predictions
                    based on signal masking for `signal_time == 0`.
                "residuals": A 2D numpy array with shape `(n_frames,
                    n_pixels_in_split)` that contains the residuals
                    based on signal masking for `signal_time == 0`.
                "mask": A 2D numpy array with shape `(width, height)`
                    that contains a binary mask indicating the subset
                    of the ROI that was processed by this function for
                    `signal_time == 0` (this mask can be different for
                    all signal times!).
            }
            "X": {
                (Same as above, but for `signal_time == X`)
            }
            ...
            "n_frames": {
                (Same as above, but for `signal_time == n_frames`)
            }
        }
        ```

        For `return_format == "full"`:

        ```
        {
            "signal_times": A numpy array with shape `(n_signal_times,)`
                containing the signal times.
            "default": {
                "predictions": A 3D numpy array (same shape as `stack`)
                    that contains the predictions for the "default"
                    case (i.e., no signal masking).
                "residuals": A 3D numpy array (same shape as `stack`)
                    that contains the residuals for the "default"
                    case (i.e., no signal masking).
            }
            "0": {
                "predictions": A 3D numpy array (same shape as `stack`)
                    that contains the predictions based on signal
                    masking for `signal_time == 0`.
                "residuals": A 3D numpy array (same shape as `stack`)
                    that contains the residuals based on signal
                    masking for `signal_time == 0`.
            }
            "X": {
                (Same as above, but for `signal_time == X`)
            }
            ...
            "n_frames": {
                (Same as above, but for `signal_time == n_frames`)
            }
        }
        ```
    """

    # -------------------------------------------------------------------------
    # Sanity checks; define shortcuts
    # -------------------------------------------------------------------------

    # Perform sanity checks on function arguments
    if return_format not in ('full', 'partial'):
        raise ValueError('return_format must be "full" or "partial"!')
    if not (0 <= roi_split < n_roi_splits):
        raise ValueError('roi_split must be in [0, n_roi_splits)!')

    # Define shortcuts
    n_frames, x_size, y_size = stack.shape
    frame_size = (x_size, y_size)
    signal_times = np.linspace(0, n_frames - 1, n_signal_times).astype(int)

    # Fit the PSF template with a 2D Gauss to get its radius
    psf_radius = get_psf_radius(psf_template=psf_template)

    # -------------------------------------------------------------------------
    # Prepare the results directory and the lookup table for indices
    # -------------------------------------------------------------------------

    # Initialize dictionary that will hold the results. We initialize it with
    # an array holding the signal times for which we have computed model with
    # signal masking, as well as a mask in which we keep track of the pixels
    # that we have processed with this script. This is useful if we are running
    # multiple instances of the training script in parallel, because it allows
    # us to save the data more space-efficient.
    #
    # NOTE: We do need a separate mask for every `signal_time`, because the
    #   pixels for which we can even train a model using signal masking depend
    #   on the signal time (through the `max_signal_length` parameter).
    tmp_results = dict(
        stack_shape=np.array(stack.shape),
        signal_times=signal_times,
        default=dict(
            predictions=[], residuals=[], mask=np.full(frame_size, False)
        )
    )
    for signal_time in signal_times:
        tmp_results[str(signal_time)] = dict(
            predictions=[], residuals=[], mask=np.full(frame_size, False)
        )

    # Create an auxiliary array which tells us the index that each spatial
    # position will end up at when we reshape a stack-shaped 3D array to a 2D
    # array by flattening the spatial dimensions.
    lookup_column_indices = np.arange(x_size * y_size).astype(int)
    lookup_column_indices = lookup_column_indices.reshape(frame_size)

    # Keep track of the indices of the columns we are processing
    processed_column_indices: List[int] = list()

    # -------------------------------------------------------------------------
    # Loop over positions in the ROI and process each pixel individually
    # -------------------------------------------------------------------------

    # Convert ROI mask into a list of positions
    roi_positions = get_positions_from_mask(roi_mask)

    # Loop over the subset of positions in the ROI that is defined by the
    # `roi_split` and `n_roi_splits` parameters
    for position in tqdm(roi_positions[roi_split::n_roi_splits], ncols=80):

        # ---------------------------------------------------------------------
        # Preliminaries: Define shortcuts, get indices
        # ---------------------------------------------------------------------

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
            selection_mask_config=selection_mask_config,
            base_model_creator=base_model_creator,
            n_splits=n_splits['default'],
            psf_radius=psf_radius,
        )

        # Now insert the results for this (spatial) position at the correct
        # position in the results list; "correct" meaning that if we turn this
        # list into a 2D numpy array and use the `results['mask']` to assign
        # it to a subset of a stack-shaped 3D array, everything ends up at the
        # expected position.
        for _ in ('predictions', 'residuals'):
            tmp_results['default'][_].insert(insert_idx, default_results[_])
        tmp_results['default']['mask'][position[0], position[1]] = 1

        # ---------------------------------------------------------------------
        # Compute and store the results based on masking a potential signal
        # ---------------------------------------------------------------------

        # Get results based on signal masking
        signal_masking_results = get_signal_masking_results(
            position=position,
            stack=stack,
            parang=parang,
            obscon_array=obscon_array,
            selection_mask_config=selection_mask_config,
            base_model_creator=base_model_creator,
            n_signal_times=n_signal_times,
            psf_radius=psf_radius,
            psf_template=psf_template,
            n_splits=n_splits['signal_masking'],
            max_signal_length=max_signal_length,
        )

        # Store result based on signal masking
        for signal_time, result in signal_masking_results.items():
            for _ in ('predictions', 'residuals'):
                tmp_results[str(signal_time)][_].insert(insert_idx, result[_])
            tmp_results[str(signal_time)]['mask'][position[0], position[1]] = 1

    # -------------------------------------------------------------------------
    # Loop over the (temporary) results dictionary and convert lists to arrays
    # -------------------------------------------------------------------------

    # Initialize a new results dictionary with the "correct" return type
    results: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]] = dict()

    # Loop over the temporary results dictionary
    for key, value in tmp_results.items():

        # First-level elements (i.e., `stack_shape`, `signal_times` and
        # `positions_mask`) are simply copied
        if not isinstance(value, dict):
            results[key] = np.array(value)

        # Second-level elements (i.e., the `predictions` and `residuals` in
        # "default" or "<signal_time>") are converted from lists to numpy
        # arrays. The transpose is necessary to allow reconstructing the 3D
        # `stack`-like shape from the 2D arrays using the `mask`.
        else:
            results[key] = dict()
            for _ in ('predictions', 'residuals'):
                results[key][_] = np.array(tmp_results[key][_]).T
            results[key]['mask'] = tmp_results[key]['mask']

    # -------------------------------------------------------------------------
    # Return results; convert to stack-shape first if desired
    # -------------------------------------------------------------------------

    # In case we want the results in the "partial" format, return them now
    if return_format == 'partial':
        return results

    # Otherwise, we need to reshape the results to the desired target format by
    # converting back the space-efficient 2D arrays to stack-like 3D arrays.

    # Initialize a dictionary for these reshaped results
    new_results = dict(signal_times=signal_times)
 
    # Loop over the different result groups to reshape them
    for group_name, group in results.items():

        # Loop only over second-level dictionaries:
        # Effectively, this means that `group_name` will either be "default"
        # or a signal time, and `group` will be the corresponding dictionary
        # holding the `predictions` and `residuals` arrays (which are 2D).
        if not isinstance(group, dict):
            continue
        new_results[group_name] = dict()

        # Define a shortcut to the positions mask
        mask = group['mask']

        # Create a new `stack`-like array for the predictions and add results
        new_results[group_name]['predictions'] = np.full(stack.shape, np.nan)
        new_results[group_name]['predictions'][:, mask] = group['predictions']

        # Create a new `stack`-like array for the residuals and add results
        new_results[group_name]['residuals'] = np.full(stack.shape, np.nan)
        new_results[group_name]['residuals'][:, mask] = group['residuals']

    return new_results


def get_default_results(
    position: Tuple[int, int],
    stack: np.ndarray,
    parang: np.ndarray,
    obscon_array: np.ndarray,
    psf_radius: float,
    base_model_creator: BaseModelCreator,
    selection_mask_config: Dict[str, Any],
    n_splits: int,
) -> Dict[str, np.ndarray]:
    """
    Get the default results for a given pixel, that is, the results
    *without* masking out any potential signal region.

    Args:
        position:
        stack:
        parang:
        obscon_array:
        psf_radius:
        base_model_creator:
        selection_mask_config:
        n_splits:

    Returns:
        A dictionary with keys "residuals" and "predictions" (which each
        map to a 1D numpy array of length `n_frames`) that contains the
        default results for the given `position`.
    """

    # -------------------------------------------------------------------------
    # Construct selection mask for this position
    # -------------------------------------------------------------------------

    # Define shortcuts to selection_mask_config
    annulus_width = Quantity(*selection_mask_config['annulus_width'])
    radius_position = Quantity(*selection_mask_config['radius_position'])
    radius_mirror_position = Quantity(
        *selection_mask_config['radius_mirror_position']
    )
    subsample_predictors = selection_mask_config['subsample_predictors']
    dilation_size = selection_mask_config['dilation_size']

    # Actually compute the selection mask
    selection_mask = get_selection_mask(
        mask_size=stack.shape[1:],
        position=position,
        signal_time=None,
        parang=parang,
        annulus_width=annulus_width,
        radius_position=radius_position,
        radius_mirror_position=radius_mirror_position,
        subsample_predictors=subsample_predictors,
        psf_radius=psf_radius,
        dilation_size=dilation_size,
        use_field_rotation=False,
    )

    # -------------------------------------------------------------------------
    # Select (and augment) predictors and targets for models
    # -------------------------------------------------------------------------

    # Select the full targets and predictors for the current position
    # using the given selection mask
    full_predictors = stack[:, selection_mask]
    full_targets = stack[:, position[0], position[1]].reshape(-1, 1)

    # Add observing conditions to the predictors
    full_predictors = np.hstack((full_predictors, obscon_array))

    # -------------------------------------------------------------------------
    # Prepare train/test split and loop over splits to train models
    # -------------------------------------------------------------------------

    # Create splitter for indices
    splitter = TrainTestSplitter(n_splits=n_splits, split_type='even_odd')

    # Prepare array for predictions
    full_predictions = np.full(len(full_targets), np.nan)

    # Loop over splits
    for train_idx, apply_idx in splitter.split(len(full_targets)):

        # Apply a scaler to the predictors and targets
        predictors_scaler = StandardScaler()
        train_predictors = predictors_scaler.fit_transform(
            full_predictors[train_idx]
        )
        apply_predictors = predictors_scaler.transform(
            full_predictors[apply_idx]
        )
        targets_scaler = StandardScaler()
        train_targets = targets_scaler.fit_transform(full_targets[train_idx])

        # Instantiate a new model
        model = base_model_creator.get_model_instance()

        # Fit the model to the data
        model.fit(train_predictors, train_targets)

        # Get the model predictions
        predictions = model.predict(apply_predictors)

        # Undo the normalization
        predictions = targets_scaler.inverse_transform(predictions).ravel()

        # Store the result
        full_predictions[apply_idx] = predictions

    # Compute full residuals
    full_residuals = full_targets.ravel() - full_predictions.ravel()

    return dict(predictions=full_predictions, residuals=full_residuals)


def get_signal_masking_results(
    position: Tuple[int, int],
    stack: np.ndarray,
    parang: np.ndarray,
    obscon_array: np.ndarray,
    selection_mask_config: Dict[str, Any],
    base_model_creator: BaseModelCreator,
    n_signal_times: int,
    psf_radius: float,
    psf_template: np.ndarray,
    n_splits: int,
    max_signal_length: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Get the results based on signal masking for given pixel: For
    `n_signal_times` points in time, compute the expected length (in
    time) of a planet signal here, mask out the corresponding temporal
    region and train a model on the rest of the time series.

    Args:
        position:
        stack:
        parang:
        obscon_array:
        selection_mask_config:
        base_model_creator:
        n_signal_times:
        psf_radius:
        psf_template:
        n_splits:
        max_signal_length:

    Returns:

    """

    # -------------------------------------------------------------------------
    # Define various shortcuts
    # -------------------------------------------------------------------------

    # Number of frames and frame size
    n_frames, *frame_size = stack.shape

    # Define shortcuts to selection_mask_config
    annulus_width = Quantity(*selection_mask_config['annulus_width'])
    radius_position = Quantity(*selection_mask_config['radius_position'])
    radius_mirror_position = Quantity(
        *selection_mask_config['radius_mirror_position']
    )
    subsample_predictors = selection_mask_config['subsample_predictors']
    dilation_size = selection_mask_config['dilation_size']

    # -------------------------------------------------------------------------
    # Prepare train/test split and loop over splits to train models
    # -------------------------------------------------------------------------

    # Initialize results dictionary
    results: Dict[str, Dict[str, Union[np.ndarray, int]]] = dict()

    # Create splitter for indices
    splitter = TrainTestSplitter(n_splits=n_splits, split_type='even_odd')

    # Prepare array for predictions
    full_predictions = np.full(n_frames, np.nan)

    # Loop over different possible planet times and exclude them from the
    # training data to find the "best" exclusion region, and thus the best
    # model, which (ideally) was trained only on the part of the time series
    # that does not contain any planet signal.
    for signal_mask, signal_time in get_signal_masks(
        position=position,
        parang=parang,
        n_signal_times=n_signal_times,
        frame_size=frame_size,
        psf_template=psf_template,
        max_signal_length=max_signal_length,
    ):

        # Define the selection mask
        selection_mask = get_selection_mask(
            mask_size=stack.shape[1:],
            position=position,
            signal_time=signal_time,
            parang=parang,
            annulus_width=annulus_width,
            radius_position=radius_position,
            radius_mirror_position=radius_mirror_position,
            subsample_predictors=subsample_predictors,
            psf_radius=psf_radius,
            dilation_size=dilation_size,
            use_field_rotation=True,
        )

        # Select the full targets and predictors for the current position
        # using the given selection mask
        full_predictors = stack[:, selection_mask]
        full_targets = stack[:, position[0], position[1]].reshape(-1, 1)

        # Add observing conditions to the predictors
        full_predictors = np.hstack((full_predictors, obscon_array))

        # Loop over cross-validation splits to avoid overfitting
        for train_idx, apply_idx in splitter.split(n_frames):

            # Construct binary versions of the train_idx and apply_idx and
            # mask out the signal region from both of them
            binary_train_idx = np.full(n_frames, False)
            binary_train_idx[train_idx] = True
            binary_train_idx[signal_mask] = False
            binary_apply_idx = np.full(n_frames, False)
            binary_apply_idx[apply_idx] = True
            binary_apply_idx[signal_mask] = False

            # Select predictors and targets for training: Choose the training
            # positions without the presumed signal region
            train_predictors = full_predictors[binary_train_idx]
            train_targets = full_targets[binary_train_idx]

            # Apply a scaler to the predictors
            predictors_scaler = StandardScaler()
            train_predictors = predictors_scaler.fit_transform(
                train_predictors
            )
            full_predictors_transformed = predictors_scaler.transform(
                full_predictors
            )

            # Apply a predictor to the targets
            targets_scaler = StandardScaler()
            train_targets = targets_scaler.fit_transform(train_targets)

            # Instantiate a new model for learning the noise
            model = base_model_creator.get_model_instance()

            # Fit the model to the training data
            model.fit(X=train_predictors, y=train_targets.ravel())

            # Get the predictions for every point in time using the current
            # model, and undo the target normalization
            predictions = model.predict(X=full_predictors_transformed)
            predictions = targets_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            )

            # Select the predictions on the "apply region" (including the
            # signal region) and store them at the right positions
            full_predictions[apply_idx] = predictions[apply_idx].ravel()

            # Check for overfitting: If the standard deviation of the residuals
            # in the train region is much smaller than the standard deviation
            # of the residuals in the apply region (without the signal region),
            # then this could be an indication that our model is memorizing the
            # training data.
            train_residuals = (full_targets - predictions)[binary_train_idx]
            apply_residuals = (full_targets - predictions)[binary_apply_idx]
            if 3 * np.std(train_residuals) < np.std(apply_residuals):
                print(f'WARNING: Seeing signs of overfitting at {position}!')

        # Compute the full residuals
        full_residuals = full_targets.ravel() - full_predictions.ravel()

        # Add predictions and residuals to results dictionary
        results[str(signal_time)] = {}
        results[str(signal_time)]['predictions'] = full_predictions.ravel()
        results[str(signal_time)]['residuals'] = full_residuals

    return results
