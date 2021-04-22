"""
Utility functions training half-sibling regression models.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from bisect import insort_left
from typing import Any, cast, Dict, List, Optional, Tuple, Union

from astropy.units import Quantity
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

import numpy as np

from hsr4hci.base_models import BaseModelCreator
from hsr4hci.forward_modeling import get_time_series_for_position
from hsr4hci.masking import get_selection_mask, get_positions_from_mask
from hsr4hci.signal_masking import get_signal_times
from hsr4hci.splitting import AlternatingSplit


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def train_all_models(
    roi_mask: np.ndarray,
    stack: np.ndarray,
    parang: np.ndarray,
    obscon_array: np.ndarray,
    selection_mask_config: Dict[str, Any],
    base_model_creator: BaseModelCreator,
    n_signal_times: int,
    psf_template: np.ndarray,
    n_splits: int,
    mode: str,
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
        mode:
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
                "residuals": A 2D numpy array with shape `(n_frames,
                    n_pixels_in_split)` that contains the residuals
                    for the "default" case (i.e., no signal masking).
                "mask": A 2D numpy array with shape `(width, height)`
                    that contains a binary mask indicating the subset
                    of the ROI that was processed by this function.
                    This mask can be used to insert the `residuals` at
                    the correct positions in a `stack`-shaped array.
            }
            "0": {
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
                "residuals": A 3D numpy array (same shape as `stack`)
                    that contains the residuals for the "default"
                    case (i.e., no signal masking).
            }
            "0": {
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
    if not 0 <= roi_split < n_roi_splits:
        raise ValueError('roi_split must be in [0, n_roi_splits)!')

    # Define shortcuts
    n_frames, x_size, y_size = stack.shape
    frame_size = (x_size, y_size)
    signal_times = get_signal_times(n_frames, n_signal_times)

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
    #   pixels for which we can even train a model using signal masking may
    #   depend on the signal time (because for signal masking, we have a
    #   threshold for the minimum fraction / amount of training data).
    tmp_results: Dict[
        str, Union[np.ndarray, Dict[str, Union[List, np.ndarray]]]
    ] = dict(
        stack_shape=np.array(stack.shape),
        signal_times=signal_times,
        default=dict(residuals=[], mask=np.full(frame_size, False)),
    )
    for signal_time in signal_times:
        tmp_results[str(signal_time)] = dict(
            residuals=[], mask=np.full(frame_size, False)
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

    # Loop over the subset of positions in the ROI (or a subset of the ROI).
    # Note: The coordinates `position` will be in the numpy convention.
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
        # Get the "default" results (= "no planet" assumption)
        # ---------------------------------------------------------------------

        # Get default results
        residuals, _ = train_model(
            stack=stack,
            parang=parang,
            obscon_array=obscon_array,
            position=position,
            mode=None,
            signal_time=None,
            expected_signal=None,
            selection_mask_config=selection_mask_config,
            psf_template=psf_template,
            n_splits=n_splits,
            base_model_creator=base_model_creator,
        )

        # Now insert the results for this (spatial) position at the correct
        # position in the results list; "correct" meaning that if we turn this
        # list into a 2D numpy array and use the `results['mask']` to assign
        # it to a subset of a stack-shaped 3D array, everything ends up at the
        # expected position.
        cast(list, tmp_results['default']['residuals']).insert(
            insert_idx, residuals
        )
        cast(np.ndarray, tmp_results['default']['mask'])[
            position[0], position[1]
        ] = 1

        # ---------------------------------------------------------------------
        # Get results based on masking / fitting a potential signal
        # ---------------------------------------------------------------------

        # Loop over different possible signal times and train models
        for signal_time in signal_times:

            # Compute expected signal based on position and signal_time
            expected_signal = get_time_series_for_position(
                position=position,
                signal_time=signal_time,
                frame_size=frame_size,
                parang=parang,
                psf_template=psf_template,
            )

            # Get results based on signal masking
            residuals, _ = train_model(
                stack=stack,
                parang=parang,
                obscon_array=obscon_array,
                position=position,
                mode=mode,
                signal_time=signal_time,
                expected_signal=expected_signal,
                selection_mask_config=selection_mask_config,
                psf_template=psf_template,
                n_splits=n_splits,
                base_model_creator=base_model_creator,
            )

            cast(list, tmp_results[str(signal_time)]['residuals']).insert(
                insert_idx, residuals
            )
            cast(np.ndarray, tmp_results[str(signal_time)]['mask'])[
                position[0], position[1]
            ] = 1

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

        # Second-level elements (i.e., the `residuals` in "default" or
        # "<signal_time>") are converted from lists to numpy arrays. The
        # transpose is necessary to allow reconstructing the 3D
        # `stack`-like shape from the 2D arrays using the `mask`.
        else:
            results[key] = dict()
            results[key]['residuals'] = np.array(
                tmp_results[key]['residuals']
            ).T
            results[key]['mask'] = np.array(tmp_results[key]['mask'])

    # -------------------------------------------------------------------------
    # Return results; convert to stack-shape first if desired
    # -------------------------------------------------------------------------

    # In case we want the results in the "partial" format, return them now
    if return_format == 'partial':
        return results

    # Otherwise, we need to reshape the results to the desired target format by
    # converting back the space-efficient 2D arrays to stack-like 3D arrays.

    # Initialize a dictionary for these reshaped results
    new_results: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]] = dict(
        signal_times=signal_times
    )

    # Loop over the different result groups to reshape them
    for group_name, group in results.items():

        # Loop only over second-level dictionaries:
        # Effectively, this means that `group_name` will either be "default"
        # or a signal time, and `group` will be the corresponding dictionary
        # holding the `residuals` array (which are 2D).
        if not isinstance(group, dict):
            continue
        new_results[group_name] = dict()

        # Define a shortcut to the positions mask
        mask = group['mask']

        # Create a new `stack`-like array for the residuals and add results
        new_results[group_name]['residuals'] = np.full(stack.shape, np.nan)
        new_results[group_name]['residuals'][:, mask] = group['residuals']

    return new_results


def train_model(
    stack: np.ndarray,
    parang: np.ndarray,
    obscon_array: np.ndarray,
    position: Tuple[int, int],
    mode: Optional[str],
    signal_time: Optional[int],
    expected_signal: Optional[np.ndarray],
    selection_mask_config: Dict[str, Any],
    psf_template: np.ndarray,
    n_splits: int,
    base_model_creator: BaseModelCreator,
    signal_masking_threshold: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Train a set of models (using a cross validation-like splitting
    scheme with `n_splits` splits) for a given spatial `position`
    and a given `signal_time`, and return the residuals and model
    parameters.

    Args:
        stack:
        parang:
        obscon_array:
        position:
        mode:
        signal_time:
        expected_signal:
        selection_mask_config:
        psf_template:
        n_splits:
        base_model_creator:
        signal_masking_threshold:

    Returns:

    """

    # -------------------------------------------------------------------------
    # Preliminaries: define a few useful shortcuts; run sanity checks
    # -------------------------------------------------------------------------

    # Define a few useful shortcuts
    n_frames = stack.shape[0]
    frame_size = (stack.shape[1], stack.shape[2])
    x, y = position

    # Make sure that `mode`, `signal_time` and `expected_signal` are either all
    # None (this is the "default" case where we basically train a model under
    # the assumption that there is no planet signal at `position`) or are all
    # not None (in case we are training with signal masking or signal fitting).
    if not (
        (
            (mode is None)
            and (signal_time is None)
            and (expected_signal is None)
        )
        or (
            (mode is not None)
            and (signal_time is not None)
            and (expected_signal is not None)
        )
    ):
        raise ValueError(
            'Invalid combination of mode, signal_time and expected_signal! '
            'Either all three must be None, or all three must be not None!'
        )

    # -------------------------------------------------------------------------
    # Construct the (spatial) selection mask for choosing the predictor pixels
    # -------------------------------------------------------------------------

    # Define shortcuts to selection_mask_config
    annulus_width = Quantity(*selection_mask_config['annulus_width'])
    radius_position = Quantity(*selection_mask_config['radius_position'])
    radius_mirror_position = Quantity(
        *selection_mask_config['radius_mirror_position']
    )

    # Define the selection mask
    # Note: get_selection_mask() expects the position to be in the astropy
    # coordinate convention, but `position` (since it is usually produced by
    # get_positions_from_mask()) is in numpy coordinates; therefore we need
    # to flip it.
    selection_mask = get_selection_mask(
        mask_size=frame_size,
        position=position[::-1],
        signal_time=signal_time,
        parang=parang,
        annulus_width=annulus_width,
        radius_position=radius_position,
        radius_mirror_position=radius_mirror_position,
        psf_template=psf_template,
    )

    # Compute the number of predictor *pixels* (since we might still add the
    # observing conditions, this is not necessarily the number of predictors)
    n_predictor_pixels = int(np.sum(selection_mask))

    # -------------------------------------------------------------------------
    # Select targets and pixel predictors; add observing conditions
    # -------------------------------------------------------------------------

    # Select the full targets and predictors for the current position
    full_predictors = stack[:, selection_mask]
    full_targets = stack[:, x, y].reshape(-1, 1)

    # Add observing conditions to the predictors
    full_predictors = np.hstack((full_predictors, obscon_array))

    # -------------------------------------------------------------------------
    # Prepare result variables
    # -------------------------------------------------------------------------

    # Prepare arrays for predictions and residuals
    full_predictions = np.full(n_frames, np.nan)
    full_residuals = np.full(n_frames, np.nan)

    # Keep track of several model parameters
    alphas = np.full(n_splits, np.nan)
    pixel_coefs = np.full((n_splits, n_predictor_pixels), np.nan)
    planet_coefs = np.full(n_splits, np.nan)

    # -------------------------------------------------------------------------
    # Prepare mask to ignore signal OR prepare signal as additional predictor
    # -------------------------------------------------------------------------

    # The default choice for the signal mask (= sample_weights) corresponds
    # to "use all available training data equally".
    signal_mask = np.ones_like(full_targets)

    # This part is only relevant if the mode / signal_time / expected_signal
    # are not None, that is, if we are training with signal masking / fitting
    if expected_signal is not None:

        # In "signal masking" mode, we ignore the part of the training data
        # which, according to the `expected_signal` time series, contains a
        # significant amount of signal. We realize this by thresholding the
        # expected signal to obtain a binary mask (where 0 = contains signal,
        # 1 = does not contain signal) which we can use as a `sample_weight`
        # when fitting the model.
        if mode == 'signal_masking':

            # The value of 0.2 as a threshold is of course somewhat arbitrary.
            # We have to use "<" because we want the mask to be True for all
            # frames that do NOT contain signal (= can be used for training).
            signal_mask = expected_signal < 0.2

            # Check if the signal mask excludes more than a given fraction of
            # the training data (again, this threshold is somewhat arbitrary).
            # In this case, we cannot train a model, and we immediately return
            # the default result values.
            if np.mean(signal_mask) < signal_masking_threshold:
                return (
                    full_residuals,
                    dict(
                        alphas=alphas,
                        pixel_coefs=pixel_coefs,
                        planet_coefs=planet_coefs,
                        selection_mask=selection_mask,
                    ),
                )

        # In "signal fitting" mode, we add the expected signal as an additional
        # predictor to a (linear) model. Usually, we are using regularized
        # models, such as ridge regression, and choose the regularization
        # strength via cross-validation. In this case, we do not want the
        # coefficient that belongs to the planet signal (i.e., the expected
        # signal) to affect the choice of the regularization strength, or
        # rather, we do not want the model to choose a "too small" coefficient
        # for the planet signal because of the regularization. A simple (but
        # somewhat hacky...) solution is to multiply the expected signal with
        # a large number, meaning that the corresponding coefficient can be
        # small (compared to the "noise part" of the model) and will thus have
        # negligible influence on the regularization term of the loss function.
        elif mode == 'signal_fitting':
            expected_signal = expected_signal.reshape(-1, 1)
            expected_signal /= np.max(expected_signal)
            expected_signal *= 1_000_000

        else:
            raise ValueError(
                'Illegal value for mode! Must be "signal_masking" or'
                '"signal_fitting"!'
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

    splitter = AlternatingSplit(n_splits=n_splits)

    for i, (train_idx, apply_idx) in enumerate(splitter.split(full_targets)):

        # ---------------------------------------------------------------------
        # Select and prepare (i.e., scale) training data
        # ---------------------------------------------------------------------

        # Select predictors, targets and sample weights for training
        train_predictors = full_predictors[train_idx]
        apply_predictors = full_predictors[apply_idx]
        train_targets = full_targets[train_idx]
        sample_weight = signal_mask[train_idx].ravel()

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

        # Instantiate a new model
        model: Any = base_model_creator.get_model_instance()

        # The following part is split based on the "mode"; this leads to some
        # code duplication, but should make the control flow more clear.

        # Case 1: We are training a "default" model ("no planet" assumption)
        if mode is None:

            # Fit the model to the (full) training data
            try:
                model.fit(X=train_predictors, y=train_targets)
            except np.linalg.LinAlgError:
                return (
                    full_residuals,
                    dict(
                        alphas=alphas,
                        pixel_coefs=pixel_coefs,
                        planet_coefs=planet_coefs,
                        selection_mask=selection_mask,
                    ),
                )

        # Case 2: We are training the model with "signal masking"
        elif mode == 'signal_masking':

            # Fit the model to the training data selected by the sample_weight
            try:
                model.fit(
                    X=train_predictors,
                    y=train_targets,
                    sample_weight=sample_weight,
                )
            except np.linalg.LinAlgError:
                return (
                    full_residuals,
                    dict(
                        alphas=alphas,
                        pixel_coefs=pixel_coefs,
                        planet_coefs=planet_coefs,
                        selection_mask=selection_mask,
                    ),
                )

        # Case 3: We are training the model with "signal fitting"
        elif mode == 'signal_fitting' and expected_signal is not None:

            # Note: the extra "and" clause is necessary for mypy; if we have
            # gotten to this point, it should always be true.

            # Add expected signal to the train predictors
            # Note that we do this *after* the "regular" `train_predictors`
            # have already been normalized! The "signal predictor" (i.e.,
            # the expected signal) is NOT normalized again!
            train_predictors_ = np.column_stack(
                [train_predictors, expected_signal[train_idx]]
            )

            # Fit the model to the (full) training data, including the extra
            # predictor in form of the expected signal
            try:
                model.fit(X=train_predictors_, y=train_targets)
            except np.linalg.LinAlgError:
                return (
                    full_residuals,
                    dict(
                        alphas=alphas,
                        pixel_coefs=pixel_coefs,
                        planet_coefs=planet_coefs,
                        selection_mask=selection_mask,
                    ),
                )

            # Ideally, we would constrain the coefficient of the expected
            # signal (and ONLY this coefficient) to be non-negative. After
            # all, there is no such thing as a "negative planet". However,
            # such a constrained model does not have an analytic solution
            # anymore (unlike "normal" linear models) and can only be learned
            # used optimization / quadratic programming, which would require
            # a custom model class (sklearn do not provide linear models where
            # you can place constraints on individual coefficients) and also
            # increase training time.
            # For these reasons, we use the following simple (and, again,
            # somewhat hacky...) solution: We simply  check if the coefficient
            # that belongs to the expected signal is negative. In this case,
            # we train the model again, this time WITHOUT the expected signal
            # as a predictor (effectively forcing the coefficient to 0).
            if model.coef_[-1] < 0:
                try:
                    model.fit(X=train_predictors, y=train_targets)
                except np.linalg.LinAlgError:
                    return (
                        full_residuals,
                        dict(
                            alphas=alphas,
                            pixel_coefs=pixel_coefs,
                            planet_coefs=planet_coefs,
                            selection_mask=selection_mask,
                        ),
                    )

            # If the coefficient that belongs to the expected_signal was NOT
            # negative, we can create a "noise only"-model by simply dropping
            # the last coefficient from the model.
            else:
                planet_coefs[i] = float(model.coef_[-1])
                model.coef_ = model.coef_[:-1]

        # All other cases result in an error (this should never happen,
        # because we have already checked for this case before)
        else:
            raise ValueError

        # ---------------------------------------------------------------------
        # Apply the model to the hold-out data to get the predictions
        # ---------------------------------------------------------------------

        # Use the model learned on the `train_predictors` to get a prediction
        # on the `apply_predictors`, and undo the target normalization
        predictions = model.predict(X=apply_predictors).reshape(-1, 1)
        predictions = t_scaler.inverse_transform(predictions).ravel()

        # Store the predictions for the current split; compute residuals
        full_predictions[apply_idx] = predictions
        full_residuals[apply_idx] = (
            full_targets[apply_idx].ravel() - predictions
        ).ravel()

        # ---------------------------------------------------------------------
        # Store additional parameters / information about the model
        # ---------------------------------------------------------------------

        # Store regularization strength of this split
        if hasattr(model, 'alpha_'):
            alphas[i] = float(model.alpha_)

        # Store pixel coefficients. In the case of a linear model, this is
        # basically "the model" (up to the intercept).
        if hasattr(model, 'coef_'):
            pixel_coefs[i] = model.coef_[:n_predictor_pixels]

    # -------------------------------------------------------------------------
    # Return results (residuals and model information)
    # -------------------------------------------------------------------------

    return (
        full_residuals,
        dict(
            alphas=alphas,
            pixel_coefs=pixel_coefs,
            planet_coefs=planet_coefs,
            selection_mask=selection_mask,
        ),
    )
