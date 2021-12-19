"""
Utility functions training half-sibling regression models.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Dict, Optional, Tuple, Union

from astropy.units import Quantity
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

import numpy as np

from hsr4hci.base_models import BaseModelCreator
from hsr4hci.forward_modeling import get_time_series_for_position
from hsr4hci.masking import (
    get_predictor_pixel_selection_mask,
    get_positions_from_mask,
    get_partial_roi_mask,
)

from hsr4hci.splitting import AlternatingSplit
from hsr4hci.typehinting import RegressorModel
from hsr4hci.utils import check_consistent_size


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_signal_times(n_frames: int, n_signal_times: int) -> np.ndarray:
    """
    Simple function to generate a temporal grid of signal times; mostly
    to ensure consistency everywhere.

    Args:
        n_frames: The total number of frames in the stack.
        n_signal_times: The number of positions on the temporal grid
            that we create.

    Returns:
        A 1D numpy array of shape `(n_signal_times, )` containing the
        temporal grid (i.e., signal times) as integers.
    """

    # Generate `n_signal_times` different possible points in time (distributed
    # uniformly over the observation) at which the planet signal could be
    return np.linspace(0, n_frames - 1, n_signal_times).astype(int)


def train_all_models(
    roi_mask: np.ndarray,
    stack: np.ndarray,
    parang: np.ndarray,
    obscon_array: np.ndarray,
    selection_mask_config: Dict[str, Any],
    base_model_creator: BaseModelCreator,
    psf_template: np.ndarray,
    train_mode: str,
    n_train_splits: int = 3,
    n_signal_times: int = 30,
    n_roi_splits: int = 1,
    roi_split: int = 0,
    return_format: str = 'full',
) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
    """
    Loop over all positions selected by the `roi_mask` (or rather, the
    subset given by `roi_split` and `n_roi_splits`), train a model for
    each pixel (using `train_model_for_position()`) and each potential
    signal time from the temporal grid, and return the results formatted
    according to the requested `return_format`.

    Args:
        roi_mask: A 2D numpy array of shape `(x_size, y_size)` that
            contains a binary mask to select the region of interest,
            that is, the pixels for which to train noise models.
        stack: A 3D numpy array of shape `(n_frames, x_size, y_size)`
            containing the data on which to train the noise models.
        parang: A 1D numpy array of shape `(n_frames, )` containing the
            corresponding parallactic angle for each frame.
        obscon_array: A 2D numpy array of shape `(n_frames, n_features)`
            containing the observing conditions that should be used as
            additional predictors.
        train_mode: The mode to use for training; must be one of the
            following: "default", "signal_fitting" or "signal_masking".
        selection_mask_config: A dictionary containing two keys (namely
            "radius_position" and "radius_opposite") that define the
            mask that is used to select the predictor pixels. The values
            of the dict should be tuples of the form `(value, "unit")`.
        psf_template: A 2D numpy array containing the unsaturated PSF
            template.
        n_train_splits: The number of training / test splits to use.
        base_model_creator: An instance of `BaseModelCreator` that can
            be used to instantiate new base models.
        n_signal_times: The size of the temporal grid, that is, the
            number of different (temporal) signal positions that are
            assumed for each pixel.
        n_roi_splits: The (total) number of splits into which the ROI
            should be divided.
        roi_split: The index of the split for which to return the mask.
        return_format: The format in which the residuals are returned.
            If "full", the residuals are 3D arrays that have the same
            size as the `stack`. If "partial", the residuals are 2D
            arrays that have the shape `(n_frames, n_pixels_in_split)`.
            The latter is recommended when training in parallel, because
            otherwise we waste a *lot* of storage for storing NaNs in
            the intermediate result files.

    Returns:
        A dictionary containing three keys:
        (1) "stack_shape": the shape of the original stack; required
            when merging partial result files.
        (2) "roi_mask": the *PARTIAL* ROI mask that was used for
            training; also required when merging partial result files.
        (3) "residuals": a dictionary with keys "default", "0", ...,
            "N", where the latter are the signal times for which we have
            trained models. Each key maps onto a numpy array containing
            the residuals for the respective model.
    """

    # Run some basic sanity checks
    check_consistent_size(stack, parang, axis=0)
    if train_mode not in ('default', 'signal_fitting', 'signal_masking'):
        raise ValueError('Illegal value for train_mode!')
    if return_format not in ('full', 'partial'):
        raise ValueError('Illegal value for return_format!')
    if not (isinstance(n_roi_splits, int) and n_roi_splits > 0):
        raise ValueError('n_roi_splits must be a positive integer!')
    if not (isinstance(roi_split, int) and 0 <= roi_split < n_roi_splits):
        raise ValueError('roi_split must be an integer in [0, n_roi_splits)!')

    # Define shortcuts
    n_frames = stack.shape[0]
    signal_times = get_signal_times(n_frames, n_signal_times)

    # Get the partial ROI mask (that selects the subset of the ROI defined by
    # `roi_split` and `n_roi_splits`) and count the number of models to train
    partial_roi_mask = get_partial_roi_mask(roi_mask, roi_split, n_roi_splits)
    n_models = int((n_signal_times + 1) * np.sum(partial_roi_mask))

    # Set up a progress bar to keep track of the training progress
    with tqdm(ncols=80, total=n_models) as progressbar:

        # Initialize dictionary in which we will collect *all* residuals
        residuals: Dict[str, np.ndarray] = {}

        # Loop over both the default model and the temporal grid, train the
        # respective models, compute the residuals, and store them
        for key in ['default'] + list(signal_times):

            # Initialize temporary array for residuals for the current key
            key_residuals = np.full_like(stack, np.nan)

            # Define train mode and signal time
            train_mode_ = 'default' if key == 'default' else train_mode
            signal_time = None if key == 'default' else int(key)

            # Loop over the pixels in the ROI split and train a model for each
            for x, y in get_positions_from_mask(partial_roi_mask):
                residual, _ = train_model_for_position(
                    stack=stack,
                    parang=parang,
                    obscon_array=obscon_array,
                    position=(x, y),
                    train_mode=train_mode_,
                    signal_time=signal_time,
                    selection_mask_config=selection_mask_config,
                    psf_template=psf_template,
                    n_train_splits=n_train_splits,
                    base_model_creator=base_model_creator,
                )
                key_residuals[:, x, y] = residual
                progressbar.update()

            # Store the residuals either in the full or partial format
            if return_format == 'full':
                residuals[str(key)] = key_residuals
            elif return_format == 'partial':
                residuals[str(key)] = key_residuals[:, partial_roi_mask]

    # Return the residuals together with the stack shape and the mask for
    # the (subset of) the ROI that we processed here
    return {
        'stack_shape': np.array(stack.shape),
        'roi_mask': partial_roi_mask,
        'residuals': residuals,
    }


def train_model_for_position(
    stack: np.ndarray,
    parang: np.ndarray,
    obscon_array: np.ndarray,
    position: Tuple[int, int],
    train_mode: str,
    signal_time: Optional[int],
    selection_mask_config: Dict[str, Any],
    psf_template: np.ndarray,
    n_train_splits: int,
    base_model_creator: BaseModelCreator,
    expected_signal: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Train a model (or rather: a set of models, because of the train /
    test splitting scheme) for a given position.

    Args:
        stack: A 3D numpy array of shape `(n_frames, x_size, y_size)`
            containing the data on which to train the noise models.
        parang: A 1D numpy array of shape `(n_frames, )` containing the
            corresponding parallactic angle for each frame.
        obscon_array: A 2D numpy array of shape `(n_frames, n_features)`
            containing the observing conditions that should be used as
            additional predictors.
        position: A tuple `(x, y)` of integers containing the position
            for which to train the model(s).
        train_mode: The mode to use for training; must be one of the
            following: "default", "signal_fitting" or "signal_masking".
        signal_time: If `train_mode` is "default", this should be None.
            Otherwise, this should contain the time at which the planet
            signal is assumed to peak at the given `position` (we need
            this value to be able to compute the forward model for
            signal fitting / masking).
        selection_mask_config: A dictionary containing two keys (namely
            "radius_position" and "radius_opposite") that define the
            mask that is used to select the predictor pixels. The values
            of the dict should be tuples of the form `(value, "unit")`.
        psf_template: A 2D numpy array containing the unsaturated PSF
            template.
        n_train_splits: The number of training / test splits to use.
        base_model_creator: An instance of `BaseModelCreator` that can
            be used to instantiate new base models.
        expected_signal: If the `train_mode` is signal fitting or signal
            masking, you can *optionally* also pass the expected signal
            explicitly to this function to avoid computing it here.
            This option may be useful when the HSR is used "hypothesis-
            based" instead of for a blind search, that is, we already
            have a hypothesis about the planet position from which we
            can compute the expected signal stack, meaning we do not
            need to loop over a temporal grid but only train a single
            model per pixel (either "default" or "signal_fitting" /
            "signal_masking").
            Note that the `expected_signal` should be consistent with
            the given `signal_time`; otherwise the mask that is used for
            the pixel predictor selection will be wrong.

    Returns:
        A 2-tuple consisting of:
        (1) the residual time series for the given `position`,
        (2) a dictionary containing additional debugging information
            about the model(s) that we have trained; for example, the
            values of the coefficients or regularization coefficients.
    """

    # -------------------------------------------------------------------------
    # Construct predictor pixel selection mask; select predictor pixels
    # -------------------------------------------------------------------------

    # Define a few useful shortcuts
    n_frames = stack.shape[0]
    frame_size = (stack.shape[1], stack.shape[2])

    # Construct the selection mask for the pixel predictors.
    # Note: `get_predictor_pixel_selection_mask()` expects the position to be
    # in the astropy coordinate convention, but `position` (since it is usually
    # produced by `get_positions_from_mask()`) is in numpy coordinates;
    # therefore we need to flip it.
    selection_mask = get_predictor_pixel_selection_mask(
        mask_size=frame_size,
        position=position[::-1],
        radius_position=Quantity(*selection_mask_config['radius_position']),
        radius_opposite=Quantity(*selection_mask_config['radius_opposite']),
        radius_excluded=Quantity(*selection_mask_config['radius_excluded']),
    )

    # Compute the number of predictor *pixels* (since we might still add the
    # observing conditions, this is not necessarily the number of predictors)
    n_pred_pixels = int(np.sum(selection_mask))

    # Select the full targets and predictors for the current position
    full_predictors = stack[:, selection_mask]
    full_targets = stack[:, position[0], position[1]].reshape(-1, 1)

    # Add observing conditions to the predictors
    # Note: the obscon_array can be empty (i.e., shape == (n_frames, 0))
    full_predictors = np.hstack((full_predictors, obscon_array))

    # -------------------------------------------------------------------------
    # Prepare result variables
    # -------------------------------------------------------------------------

    # Prepare array for predictions
    full_predictions = np.full(n_frames, np.nan)

    # Keep track of several model parameters
    alphas = np.full(n_train_splits, np.nan)
    pixel_coefs = np.full((n_train_splits, n_pred_pixels), np.nan)
    planet_coefs = np.full(n_train_splits, np.nan)

    # -------------------------------------------------------------------------
    # Compute the expected signal (if needed)
    # -------------------------------------------------------------------------

    # If we have already received an expected_signal (e.g., because we are
    # running the HSR in "hypothesis-based mode"), we do not compute it here
    if expected_signal is None:

        # Always initialize the expected signal
        expected_signal = np.full(n_frames, np.nan)

        # Only compute it if we are not training a default model. This happens
        # here, so we don't have to re-compute it in each train / test-split.
        if train_mode in ('signal_fitting', 'signal_masking'):

            # Ensure that the signal time is not None
            if signal_time is None:
                raise RuntimeError('signal_time must not be None!')

            # Compute expected signal based on position and signal_time. The
            # resulting time series is already normalized to a maximum of 1.
            expected_signal = get_time_series_for_position(
                position=position,
                signal_time=signal_time,
                frame_size=frame_size,
                parang=parang,
                psf_template=psf_template,
            )

    # -------------------------------------------------------------------------
    # Train model(s)
    # -------------------------------------------------------------------------

    # To prevent overfitting, we do not simply fit a model to the full data
    # and then get its prediction (i.e., the fit result). Instead, we split
    # the data into "training" and "application" (in a cross-validation way)
    # and learn models on the training data before applying them (to get the
    # prediction) to the application data.
    # This ensures that the model cannot simply memorize the "correct"
    # prediction, which otherwise can happen even for simple models in cases
    # where we have more predictors than time steps.

    splitter = AlternatingSplit(n_splits=n_train_splits)

    for i, (train_idx, apply_idx) in enumerate(splitter.split(full_targets)):

        # ---------------------------------------------------------------------
        # Select and prepare (i.e., scale / whiten) the training data
        # ---------------------------------------------------------------------

        # Select predictors, targets and sample weights for training
        train_predictors = full_predictors[train_idx]
        apply_predictors = full_predictors[apply_idx]
        train_targets = full_targets[train_idx]

        # Apply a scaler to the predictors
        p_scaler = StandardScaler()
        train_predictors = p_scaler.fit_transform(train_predictors)
        apply_predictors = p_scaler.transform(apply_predictors)

        # Apply a predictor to the targets
        t_scaler = StandardScaler()
        train_targets = t_scaler.fit_transform(train_targets).ravel()

        # ---------------------------------------------------------------------
        # Train the model (which, in the end, should only predict noise!)
        # ---------------------------------------------------------------------

        # Either train a default model...
        if train_mode == 'default':
            model = _train_default_model(
                base_model_creator=base_model_creator,
                train_predictors=train_predictors,
                train_targets=train_targets,
            )

        # ... or a model with signal fitting / fitting
        elif train_mode == 'signal_fitting':
            model, planet_coefficient = _train_signal_fitting_model(
                base_model_creator=base_model_creator,
                train_predictors=train_predictors,
                train_targets=train_targets,
                expected_signal=expected_signal[train_idx],
            )
            planet_coefs[i] = planet_coefficient

        # ... or a model with signal masking
        elif train_mode == 'signal_masking':
            model = _train_signal_masking_model(
                base_model_creator=base_model_creator,
                train_predictors=train_predictors,
                train_targets=train_targets,
                expected_signal=expected_signal[train_idx],
            )

        else:
            raise ValueError('Illegal value for train_mode!')

        # ---------------------------------------------------------------------
        # Apply the model to the hold-out data to get the predictions
        # ---------------------------------------------------------------------

        # If the model is None, training did not succeed for some reason (for
        # example, the signal masked would have excluded a too large fraction
        # of the data). In this case, we do not get any predictions.
        # Otherwise---that if the training succeeded---we can apply the model
        # to the data that were held out by the split to get the predictions.
        if model is not None:

            # Use the model (that we learned on the `train_predictors`) to
            # get a prediction on the `apply_predictors`, and undo the target
            # normalization
            predictions = model.predict(X=apply_predictors).reshape(-1, 1)
            predictions = t_scaler.inverse_transform(predictions).ravel()

            # Store the predictions for the current split
            full_predictions[apply_idx] = predictions

            # For regularized models: store the regularization strength of
            # this split (for debugging purposes)
            if hasattr(model, 'alpha_'):
                if np.isscalar(model.alpha_):  # type: ignore
                    alphas[i] = float(model.alpha_)  # type: ignore

            # For linear models: store pixel coefficients. In the case of a
            # linear model, this is basically the model (up to the intercept).
            if hasattr(model, 'coef_'):
                pixel_coefs[i] = model.coef_[:n_pred_pixels]  # type: ignore

    # -------------------------------------------------------------------------
    # Compute residuals and return results
    # -------------------------------------------------------------------------

    # After loop over all train/test splits is complete, compute residuals
    full_residuals = full_targets.ravel() - full_predictions.ravel()

    # Finally, return the residuals and the information about the model
    return (
        full_residuals,
        dict(
            alphas=alphas,
            pixel_coefs=pixel_coefs,
            planet_coefs=planet_coefs,
            selection_mask=selection_mask,
        ),
    )


def _train_default_model(
    base_model_creator: BaseModelCreator,
    train_predictors: np.ndarray,
    train_targets: np.ndarray,
) -> Optional[RegressorModel]:
    """
    Train a default model (i.e., no signal fitting or masking).

    Args:
        base_model_creator: Instance of `BaseModelCreator` that can be
            used to instantiate a new model.
        train_predictors: A 2D numpy array containing the (normalized)
            predictors. Shape: `(n_time_steps, n_features)`.
        train_targets: A 1D numpy array containing the (normalized)
            targets. Shape `(n_time_steps, )`.

    Returns:
        The trained model instance, or None, if the training failed
        with a `np.linalg.LinAlgError`.
    """

    # Instantiate a new model
    model = base_model_creator.get_model_instance()

    # Fit the model to the (full) training data
    try:
        model.fit(X=train_predictors, y=train_targets)
        return model
    except np.linalg.LinAlgError:  # pragma: no cover
        return None


def _train_signal_fitting_model(
    base_model_creator: BaseModelCreator,
    train_predictors: np.ndarray,
    train_targets: np.ndarray,
    expected_signal: np.ndarray,
) -> Tuple[Optional[RegressorModel], float]:
    """
    Train a model with signal fitting.

    Args:
        base_model_creator: Instance of `BaseModelCreator` that can be
            used to instantiate a new model.
        train_predictors: A 2D numpy array containing the (normalized)
            predictors. Shape: `(n_time_steps, n_features)`.
        train_targets: A 1D numpy array containing the (normalized)
            targets. Shape: `(n_time_steps, )`.
        expected_signal: A 1D numpy array containing the expected signal
            for the hypothesis under which we are training the current
            model. Shape: `(n_time_steps, )`.

    Returns:
        A 2-tuple, consisting of (1) the trained noise model instance
        (i.e., with the planet coefficient removed), and (2) the value
        of the planet coefficient that was removed. In case the training
        fails with a `np.linalg.LinAlgError`, return `(None, np.nan)`.
    """

    # Instantiate a new model
    model: Any = base_model_creator.get_model_instance()

    # Check that we have instantiated a linear model: signal fitting is only
    # possible if we can remove the coefficient that corresponds to the signal
    # after training. This only works for linear models and neural networks
    # with a special network architecture; this implementation only supports
    # the former.
    if 'linear_model' not in model.__module__:
        raise RuntimeError('Signal fitting only works with linear models!')

    # In "signal fitting" mode, we add the expected signal as an additional
    # predictor to a (linear) model. Usually, we are using regularized models,
    # such as ridge regression, and choose the regularization strength via
    # cross-validation. In this case, we do not want the coefficient that
    # belongs to the planet signal (i.e., the expected signal) to affect the
    # choice of the regularization strength, or rather, we do not want the
    # model to choose a "too small" coefficient for the planet signal because
    # of the regularization.
    # A simple (but somewhat hacky...) solution is to multiply the expected
    # signal with a large number, meaning that the corresponding coefficient
    # can be small (compared to the "noise part" of the model) and will thus
    # have only negligible influence on the regularization term of the loss
    # function.
    # Note: the scaling factor should also not be *too large*, because in this
    # case you get "RuntimeWarning: invalid value encountered in multiply" for
    # many pixels, and the residuals are starting to look worse.
    expected_signal_ = np.copy(expected_signal)
    expected_signal_ *= 1_000
    expected_signal_ = expected_signal_.reshape(-1, 1)

    # Add the expected signal to the train predictors. We add it as the last
    # column in the predictors-matrix, meaning that we know that we can access
    # the corresponding coefficient of the model as `model.coef_[-1]`.
    # Note also that we do this *after* the "regular" `train_predictors` have
    # already been normalized / whitened! We do *not* normalize the expected
    # signal again after the above re-scaling procedure!
    train_predictors_ = np.column_stack([train_predictors, expected_signal_])

    # Fit the model to the (full) training data, including the extra predictor
    # in form of the expected signal
    try:
        model.fit(X=train_predictors_, y=train_targets)
    except np.linalg.LinAlgError:  # pragma: no cover
        return None, np.nan

    # Ideally, we would constrain the coefficient of the expected signal (and
    # ONLY this coefficient) to be non-negative. After all, there is no such
    # thing as a "negative planet". However, such a constrained model does not
    # have an analytic solution (unlike "normal" linear models) and can only
    # be learned used optimization / quadratic programming, which would require
    # a custom model class (sklearn do not provide linear models where you can
    # place constraints on individual coefficients) and also increase training
    # time.
    # For these reasons, we use the following simple (and, again, somewhat
    # hacky...) solution: We simply  check if the coefficient that belongs to
    # the expected signal is negative. In this case, we train the model again,
    # this time WITHOUT the expected signal as a predictor (effectively forcing
    # the coefficient to 0).

    # Get the coefficient that belongs to the expected signal, and undo the
    # scaling that we applied to the expected signal due to the regularization
    planet_coefficient = float(model.coef_[-1]) * 1_000

    # If the planet coefficient is negative, re-train the model *without* the
    # expected signal as a predictor
    if planet_coefficient < 0:
        try:
            model.fit(X=train_predictors, y=train_targets)
        except np.linalg.LinAlgError:  # pragma: no cover
            return None, np.nan

    # If the planet coefficient is NOT negative, we can create a "noise
    # only"-model by simply dropping the last coefficient from the model
    else:
        model.coef_ = model.coef_[:-1]
        model.n_features_in_ -= 1

    # Return the model and the planet coefficient
    return model, planet_coefficient


def _train_signal_masking_model(
    base_model_creator: BaseModelCreator,
    train_predictors: np.ndarray,
    train_targets: np.ndarray,
    expected_signal: np.ndarray,
) -> Optional[RegressorModel]:
    """
    Train a model with signal masking.

    Args:
        base_model_creator: Instance of `BaseModelCreator` that can be
            used to instantiate a new model.
        train_predictors: A 2D numpy array containing the (normalized)
            predictors. Shape: `(n_time_steps, n_features)`.
        train_targets: A 1D numpy array containing the (normalized)
            targets. Shape: `(n_time_steps, )`.
        expected_signal: A 1D numpy array containing the expected signal
            for the hypothesis under which we are training the current
            model. Shape: `(n_time_steps, )`.

    Returns:
        The trained model instance, or None, if the training failed
        with a `np.linalg.LinAlgError`.
    """

    # Threshold the expected signal to find the time steps that we must not use
    # for training (the threshold value of 0.2 is, of course, a bit arbitrary).
    # We have to use "<" because we want the mask to be True for all steps that
    # do NOT contain signal (= *can* be used for training).
    signal_mask = expected_signal < 0.2

    # Check if the signal mask excludes more than a given fraction of the
    # training data, namely, 50% of the data (again, this threshold is rather
    # arbitrary). In this case, we cannot / should not train a model.
    if np.mean(signal_mask) < 0.5:
        return None

    # Instantiate a new model
    model = base_model_creator.get_model_instance()

    # Fit the model to the training data to which we have to apply the signal
    # mask in order to ignore all parts that contain too much planet signal
    try:
        model.fit(
            X=train_predictors[signal_mask],
            y=train_targets[signal_mask],
        )
        return model
    except np.linalg.LinAlgError:  # pragma: no cover
        return None
