"""
Functions to use for training half-sibling regression models.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Callable, Dict, Tuple, Union

from astropy.units import Quantity
from sklearn.preprocessing import StandardScaler

import numpy as np

from hsr4hci.utils.masking import get_selection_mask
from hsr4hci.utils.signal_masking import get_signal_masks
from hsr4hci.utils.splitting import TrainTestSplitter


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_default_results(
    position: Tuple[int, int],
    stack: np.ndarray,
    parang: np.ndarray,
    obscon_array: np.ndarray,
    psf_diameter: float,
    get_model_instance: Callable,
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
        psf_diameter:
        get_model_instance:
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
        psf_diameter=psf_diameter,
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
        model = get_model_instance()

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
    get_model_instance: Callable,
    n_signal_times: int,
    frame_size: Tuple[int, int],
    psf_diameter: float,
    psf_cropped: np.ndarray,
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
        get_model_instance:
        n_signal_times:
        frame_size:
        psf_diameter:
        psf_cropped:
        n_splits:
        max_signal_length:

    Returns:

    """

    # -------------------------------------------------------------------------
    # Define various shortcuts
    # -------------------------------------------------------------------------

    # Number of frames
    n_frames = len(parang)

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
        psf_cropped=psf_cropped,
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
            psf_diameter=psf_diameter,
            dilation_size=dilation_size,
            use_field_rotation=True,
        )

        # Select the full targets and predictors for the current position
        # using the given selection mask
        full_predictors = stack[:, selection_mask]
        full_targets = stack[:, position[0], position[1]].reshape(-1, 1)

        # Add observing conditions to the predictors
        full_predictors = np.hstack((full_predictors, obscon_array))

        # Add sub-dictionary in results
        results[str(signal_time)] = dict(
            signal_mask=signal_mask, signal_time=signal_time
        )

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
            model = get_model_instance()

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
        results[str(signal_time)]['predictions'] = full_predictions.ravel()
        results[str(signal_time)]['residuals'] = full_residuals

    return results
