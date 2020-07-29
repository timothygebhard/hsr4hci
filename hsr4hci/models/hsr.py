"""
Provides the basic HalfSiblingRegression class.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import os

from astropy import units
from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np

from hsr4hci.models.callbacks import BaseCollector

from hsr4hci.utils.derotating import derotate_combine
from hsr4hci.utils.fits import save_fits
from hsr4hci.utils.importing import get_member_by_name
from hsr4hci.utils.masking import (
    get_roi_mask,
    get_positions_from_mask,
    get_selection_mask,
)
from hsr4hci.utils.preprocessing import PredictorsTargetsScaler
from hsr4hci.utils.splitting import TrainTestSplitter
from hsr4hci.utils.tqdm import tqdm_joblib
from hsr4hci.utils.typehinting import RegressorModel


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------


class HalfSiblingRegression:
    """
    Wrapper class for half-sibling regression models.

    This class basically keeps together the individual models for each
    pixels, that is, it encapsulates the necessary loops over the region
    of interest and provides options for parallelization.

    Args:
        config:
        results_dir:
        stack:
        parang:
        observing_conditions: A 2D numpy array of shape (n_frames, F),
            where F is the number of features / observing conditions,
            such as the air pressure or temperature.
    """

    def __init__(
        self,
        config: dict,
        results_dir: str,
        stack: np.ndarray,
        parang: np.ndarray,
        observing_conditions: np.ndarray,
    ) -> None:

        # Store the constructor arguments
        self.config = config
        self.results_dir = results_dir
        self.stack = stack
        self.parang = parang
        self.observing_conditions = observing_conditions

        # Define additional derived variables
        self.n_frames: int = self.stack.shape[0]
        self.frame_size: Tuple[int, int] = self.stack.shape[1:3]
        self.field_rotation: units.Quantity = units.Quantity(
            np.abs(self.parang[-1] - self.parang[0]), 'degree'
        )
        self.roi_mask: np.ndarray = get_roi_mask(
            mask_size=self.frame_size,
            inner_radius=self.config['roi_mask']['inner_radius'],
            outer_radius=self.config['roi_mask']['outer_radius'],
        )
        self.train_test_splitter: TrainTestSplitter = TrainTestSplitter(
            **self.config['train_test_splitter']
        )
        self.n_splits = self.train_test_splitter.n_splits

        # Initialize callbacks and variables for their results
        self.callbacks: List[Type[BaseCollector]] = [
            get_member_by_name('hsr4hci.models.callbacks', callback_name)
            for callback_name in self.config['callbacks']
        ]
        self.callback_results: Dict[str, np.ndarray] = {
            _.name: np.full(_.shape(self.frame_size, self.n_splits), np.nan)
            for _ in self.callbacks
        }

        # Initialize variables for main training results
        self.predictions: np.ndarray = np.full(self.stack.shape, np.nan)
        self.residuals: np.ndarray = np.full(self.stack.shape, np.nan)

    def _ensure_results_dir(self) -> None:
        """
        Make sure that the results directory for this instance exists.
        """

        Path(self.results_dir).mkdir(exist_ok=True, parents=True)

    def _get_base_model_instance(self) -> RegressorModel:
        """
        Get a new instance of the base model defined in the config.

        Returns:
            An instance of a regression method (e.g., from sklearn) that
            must provide the .fit() and .predict() methods.
        """

        # Get the model class and the model parameters
        module_name = self.config['base_model']['module']
        class_name = self.config['base_model']['class']
        model_class = get_member_by_name(
            module_name=module_name, member_name=class_name
        )
        model_parameters = self.config['base_model']['parameters']

        # Augment the model parameters:
        # For RidgeCV models, we increase the number of alpha values (i.e.,
        # regularization strengths). Adding them here seems easier than adding
        # N numbers to a configuration file...
        if class_name == 'RidgeCV':
            model_parameters['alphas'] = np.geomspace(1e0, 1e6, 19)

        # Instantiate a new model of the given class with the desired params
        model: RegressorModel = model_class(**model_parameters)

        return model

    def train(self) -> None:
        """
        Train models for all positions (pixels) in the ROI.

        This function is essentially a loop over .train_position();
        either "manually", or by means of joblib.Parallel.
        """

        # Define shortcuts for accessing the config
        use_multiprocessing = self.config['multiprocessing']['enable']
        n_processes = self.config['multiprocessing']['n_processes']

        # ---------------------------------------------------------------------
        # Train models for all positions
        # ---------------------------------------------------------------------

        # Get positions in ROI as a list
        roi_positions = get_positions_from_mask(self.roi_mask)

        # Now, process train models for all positions in the ROI.
        # We can either use joblib.Parallel-based parallelization...
        if use_multiprocessing:

            # Use joblib to process the positions in parallel. 'sharedmem' is
            # required to ensure the child processes have access to the memory
            # of the main process (e.g., stack, astropy.units registry, ...).
            with tqdm_joblib(tqdm(total=len(roi_positions), ncols=80)) as _:
                with Parallel(n_jobs=n_processes, require='sharedmem') as run:
                    results = run(
                        delayed(self.train_position)(position)
                        for position in roi_positions
                    )

        # ...or "manually" loop over all positions in the ROI.
        else:

            # Sequentially loop over all positions in the ROI and train models
            results = []
            for position in tqdm(roi_positions, ncols=80):
                results.append(self.train_position(position=position))

        # ---------------------------------------------------------------------
        # Collect and combine results from individual positions
        # ---------------------------------------------------------------------

        # Once all positions in the ROI have been processed (either in
        # parallel or sequentially), combine all these individual pixel
        # results into a single "global" result
        for result in results:

            # Unpack the current result dictionary
            position = result['position']
            predictions = result['predictions']

            # Define the fancy index to select position in result arrays
            position_idx = (slice(None),) + position

            # Store predictions and results of debugging callbacks
            self.predictions[position_idx] = predictions
            for key in self.callback_results.keys():
                self.callback_results[key][position_idx] = result[key]

        # Finally, compute the residuals from predictions
        self.residuals = self.stack - self.predictions

    def train_position(
        self,
        position: Tuple[int, int]
    ) -> Dict[str, Union[Tuple[int, int], np.ndarray]]:
        """
        Train models for a single given position (pixel).

        Args:
            position: A tuple (x, y) indicating the position of the
                pixel for which to train the corresponding models.
        """

        # Get the selection_mask for this position, that is, the mask that
        # selects the (spatial) pixels to be used as predictors
        selection_mask = get_selection_mask(
            mask_size=self.frame_size,
            position=position,
            field_rotation=self.field_rotation,
            **self.config['selection_mask'],
        )

        # Initialize result variables and set up callbacks
        predictions = np.full(self.n_frames, np.nan)
        callbacks = [
            callback(n_splits=self.n_splits, selection_mask=selection_mask)
            for callback in self.callbacks
        ]

        # Loop over all train/apply splits, train models and make predictions
        for split_idx, (train_idx, apply_idx) in enumerate(
            self.train_test_splitter.split(self.n_frames)
        ):

            # Set up a new scaler for predictors and predictors
            scaler_type = self.config['preprocessing']['scaler_type']
            scaler = PredictorsTargetsScaler(scaler_type=scaler_type)

            # Select predictors and targets for both training and application
            train_predictors = self.stack[train_idx][:, selection_mask]
            apply_predictors = self.stack[apply_idx][:, selection_mask]
            train_targets = self.stack[train_idx, position[0], position[1]]
            apply_targets = self.stack[apply_idx, position[0], position[1]]

            # Add the observing conditions to the predictors (we add them at
            # this stage so that they are also standardized by the scaler)
            # TODO: This is only for the very simplest of models, for a more
            #       complicated model, we need a more general approach!
            train_predictors = np.hstack(
                (train_predictors, self.observing_conditions[train_idx])
            )
            apply_predictors = np.hstack(
                (apply_predictors, self.observing_conditions[apply_idx])
            )

            # Scale predictors and targets to bring all spatial pixels to the
            # same scale (this is only important for regularized models)
            (
                train_predictors,
                apply_predictors,
            ) = scaler.fit_transform_predictors(
                X_train=train_predictors, X_apply=apply_predictors
            )
            train_targets, apply_targets = scaler.fit_transform_targets(
                X_train=train_targets, X_apply=apply_targets
            )

            # Instantiate a new model and fit it to the training data
            model = self._get_base_model_instance()
            model.fit(X=train_predictors, y=train_targets.ravel())

            # Apply the learned model to the data that was held out and
            # store the predictions
            predictions[apply_idx] = model.predict(X=apply_predictors)

            # Apply inverse scaling transform to predictions to move them back
            # to the same scale as the data from which they will be subtracted
            predictions[apply_idx] = scaler.inverse_transform_targets(
                X=predictions[apply_idx]
            )

            # Run callbacks to collect their data
            for callback in callbacks:
                callback.collect(
                    model=model,
                    split_idx=split_idx,
                    y_true=apply_targets,
                    y_pred=predictions[apply_idx],
                )

        # Initialize results with the indispensables (position and predictions)
        results = {'position': position, 'predictions': predictions}

        # Loop over callbacks and add their data to the results
        for callback in callbacks:
            results[callback.name] = callback.get_results()

        return results

    def get_signal_estimate(
        self,
        subtract: Optional[str] = None,
        combine: str = 'mean'
    ) -> np.ndarray:
        """
        Compute the estimate for the planet signal from the residuals.

        Args:
            subtract: What to subtract from the residuals before
                derotating them (i.e., "mean" or "median").
            combine: How to combine (average) the derotated residual
                frames (i.e., "mean" or "median").

        Returns:
            A 2D numpy array containing the estimate for the planet
            signal as computed from the residuals.
        """

        # Derotate and merge residuals to compute signal estimate
        signal_estimate = derotate_combine(
            stack=self.residuals,
            parang=self.parang,
            mask=(~self.roi_mask),
            subtract=subtract,
            combine=combine,
        )

        return signal_estimate

    def get_self_subtraction_estimate(
        self,
        subtract: Optional[str] = None,
        combine: str = 'median'
    ) -> np.ndarray:
        """
        Compute the estimate for the self-subtraction from the noise
        model, that is, de-rotate and combine the noise model.

        Of course, this is quantity is only a proxy for the actual loss
        of planet signal; however, if we see a planet in this estimate,
        this is a bad sign.

        Args:
            subtract: What to subtract from the noise model before
                derotating it (i.e., "mean" or "median").
            combine: How to combine (average) the derotated noise model
                frames (i.e., "mean" or "median").

        Returns:
            A 2D numpy array containing the estimate for the self-
            subtraction as computed from the noise model.
        """

        # Derotate and merge the noise model (i.e., the stack minus the
        # residuals) to compute our estimate for the self subtraction
        self_subtraction = derotate_combine(
            stack=(self.stack - self.residuals),
            parang=self.parang,
            mask=(~self.roi_mask),
            subtract=subtract,
            combine=combine,
        )

        return self_subtraction

    def save_self_subtraction_estimate(self) -> None:
        """
        Save self-subtraction as a FITS file to the results directory.
        """

        print('Saving self-subtraction to FITS...', end=' ', flush=True)
        self._ensure_results_dir()
        file_path = os.path.join(self.results_dir, 'self_subtraction.fits')
        save_fits(
            array=self.get_self_subtraction_estimate(), file_path=file_path
        )
        print('Done!', flush=True)

    def save_predictions(self) -> None:
        """
        Save predictions as a FITS file to the results directory.
        """

        print('Saving predictions to FITS...', end=' ', flush=True)
        self._ensure_results_dir()
        file_path = os.path.join(self.results_dir, 'predictions.fits')
        save_fits(array=self.predictions, file_path=file_path)
        print('Done!', flush=True)

    def save_residuals(self) -> None:
        """
        Save residuals as a FITS file to the results directory.
        """

        print('Saving residuals to FITS...', end=' ', flush=True)
        self._ensure_results_dir()
        file_path = os.path.join(self.results_dir, 'residuals.fits')
        save_fits(array=self.residuals, file_path=file_path)
        print('Done!', flush=True)

    def save_debugging_results(self) -> None:
        """
        Save results of debugging callbacks to FITS files.
        """

        # Loop over all debugging results and save them as FITS files
        for key, value in self.callback_results.items():

            print(f'Saving {key} to FITS...', end=' ', flush=True)
            self._ensure_results_dir()
            file_path = os.path.join(self.results_dir, f'{key}.fits')
            save_fits(array=value, file_path=file_path)
            print('Done!', flush=True)
