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

from hsr4hci.models.callbacks import DebuggingCollector

from hsr4hci.utils.derotating import derotate_combine
from hsr4hci.utils.fits import save_fits
from hsr4hci.utils.importing import get_member_by_name
from hsr4hci.utils.masking import get_roi_mask, get_positions_from_mask, \
    get_selection_mask
from hsr4hci.utils.splitting import TrainTestSplitter
from hsr4hci.utils.tqdm import tqdm_joblib


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class HalfSiblingRegression:
    """
    Wrapper class for half-sibling regression models.

    This class basically keeps together the individual models for each
    pixels, that is, it encapsulates the necessary loops over the region
    of interest and provides options for parallelization.
    """

    def __init__(self,
                 config: dict,
                 results_dir: str):

        # Store the constructor arguments
        self.config = config
        self.results_dir = results_dir

        # Define a few useful shortcuts to the configuration
        self.config_model = config['base_model']
        self.experiment_dir = config['experiment_dir']
        self.frame_size = tuple(config['dataset']['frame_size'])

        # Compute implicitly defined class variables (i.e., the ROI mask)
        self.roi_mask: np.ndarray = \
            get_roi_mask(mask_size=self.frame_size,
                         inner_radius=config['roi_mask']['inner_radius'],
                         outer_radius=config['roi_mask']['outer_radius'])

        # Initialize debugging callbacks and their results storage
        self.debugging_callbacks = \
            self._initialize_debugging_callbacks(config['debugging_callbacks'])
        self.debugging_results: Dict[str, np.ndarray] = dict()

        # Initialize additional class variables
        self.field_rotation: Optional[float] = None
        self.predictions: Optional[np.ndarray] = None
        self.residuals: Optional[np.ndarray] = None
        self.train_test_splitter: Optional[TrainTestSplitter] = None

    @property
    def splitter(self) -> TrainTestSplitter:

        # Create a function to split the stack into training and test
        if self.train_test_splitter is None:
            self.train_test_splitter = \
                TrainTestSplitter(**self.config['train_test_splitter'])
        return self.train_test_splitter

    @staticmethod
    def _initialize_debugging_callbacks(list_of_callback_names: List[str]) \
            -> List[Type[DebuggingCollector]]:
        """
        Turn a list of callback class names (e.g., 'AlphaCollector')
        into a list of actual classes that can be instantiated.

        Args:
            list_of_callback_names: A list of strings specifying the
                names of the callbacks from the debugging module that
                we want to use during the training; for example:
                "AlphaCollector" for a callback that collects the
                .alpha_ parameter of the trained models.

        Returns:
            A list of classes that can be used as callbacks during the
            training of the models (must be from the debugging module).
        """

        # Initialize result list
        debugging_callbacks = list()

        # Loop over the list of callback names and import the corresponding
        # classes from the debugging module
        for callback_name in list_of_callback_names:
            debugging_callback: Type[DebuggingCollector] = \
                get_member_by_name(module_name='hsr4hci.models.debugging',
                                   member_name=callback_name)
            debugging_callbacks.append(debugging_callback)

        return debugging_callbacks

    def _ensure_results_dir(self):
        """
        Make sure that the results directory for this instance exists.
        """

        Path(self.results_dir).mkdir(exist_ok=True, parents=True)

    def _get_base_model_instance(self):
        """
        Get a new instance of the base model defined in the config.

        Returns:
            An instance of a regression method (e.g., from sklearn) that
            must provide the .fit() and .predict() methods.
        """

        # Get the model class and the model parameters
        module_name = self.config['base_model']['module']
        class_name = self.config['base_model']['class']
        model_class = get_member_by_name(module_name=module_name,
                                         member_name=class_name)
        model_parameters = self.config['base_model']['parameters']

        # Augment the model parameters:
        # For RidgeCV models, we increase the number of alpha values (i.e.,
        # regularization strengths). Adding them here seems easier than adding
        # N numbers to a configuration file...
        if class_name == 'RidgeCV':
            model_parameters['alphas'] = np.geomspace(1e0, 1e6, 19)

        # Instantiate a new model of the given class with the desired params
        model = model_class(**model_parameters)

        return model

    def _get_field_rotation(self,
                            parang: np.ndarray) -> units.Quantity:

        # If we have not computed the field rotation of the data set we are
        # using for training, do it now and store the result
        if self.field_rotation is None:
            self.field_rotation = np.abs(parang[-1] - parang[0])
            self.field_rotation = units.Quantity(self.field_rotation, 'degree')
        return self.field_rotation

    def train(self,
              stack: np.ndarray,
              parang: np.ndarray) -> None:
        """
        Train models for all positions (pixels) in the ROI.

        This function is essentially a loop over .train_position();
        either "manually", or by means of joblib.Parallel.

        Args:
            stack: A numpy array of shape (n_frames, width, height)
                containing the stack of images to be processed.
            parang: A numpy array of shape (n_frames, ) containing the
                respective parallactic angles for each image.
        """

        # Define shortcuts for accessing the config
        use_multiprocessing = self.config['multiprocessing']['enable']
        n_processes = self.config['multiprocessing']['n_processes']
        n_splits = self.config['train_test_splitter']['n_splits']

        # ---------------------------------------------------------------------
        # Train models for all positions
        # ---------------------------------------------------------------------

        # Get positions in ROI as a list
        roi_positions = get_positions_from_mask(self.roi_mask)

        # Define a partial application for self.train_position() that
        # fixes the stack and the parallactic angles
        def train_position(position):
            return self.train_position(position, stack, parang)

        # Now, process train models for all positions in the ROI.
        # We can either use joblib.Parallel-based parallelization...
        if use_multiprocessing:

            # Use joblib to process the positions in parallel. 'sharedmem' is
            # required to ensure the child processes have access to the memory
            # of the main process (e.g., stack, astropy.units registry, ...).
            with tqdm_joblib(tqdm(total=len(roi_positions), ncols=80)) as _:
                with Parallel(n_jobs=n_processes, require='sharedmem') as run:
                    results = run(delayed(train_position)(position)
                                  for position in roi_positions)

        # ...or "manually" loop over all positions in the ROI.
        else:

            # Sequentially loop over all positions in the ROI and train models
            results = []
            for position in tqdm(roi_positions, ncols=80):
                results.append(train_position(position=position))

        # ---------------------------------------------------------------------
        # Collect and combine results from individual positions
        # ---------------------------------------------------------------------

        # Initialize result variables
        self.predictions = np.full(stack.shape, np.nan)
        self.debugging_results = \
            {_.result_name: np.full(_.shape(n_splits=n_splits,
                                            frame_size=self.frame_size),
                                    np.nan) for _ in self.debugging_callbacks}

        # Once all positions in the ROI have been processed (either in
        # parallel or sequentially), combine all these individual pixel
        # results into a single "global" result
        for result in results:

            # Unpack the current result dictionary
            position = result['position']
            predictions = result['predictions']

            # Define the fancy index to select position in result arrays
            position_idx = (slice(None), ) + position

            # Store predictions and results of debugging callbacks
            self.predictions[position_idx] = predictions
            for key in self.debugging_results.keys():
                self.debugging_results[key][position_idx] = result[key]

        # Finally, compute the residuals from predictions
        self.residuals = stack - self.predictions

    def train_position(self,
                       position: Tuple[int, int],
                       stack: np.ndarray,
                       parang: np.ndarray) -> Dict[str, Union[Tuple[int, int],
                                                              np.ndarray]]:
        """
        Train models for a single given position (pixel).

        Args:
            position: A tuple (x, y) indicating the position of the
                pixel for which to train the corresponding models.
            stack: A numpy array of shape (n_frames, width, height)
                containing the stack of images to be processed.
            parang: A numpy array of shape (n_frames, ) containing the
                respective parallactic angles for each image.
        """

        # Define shortcuts for number of frames
        n_frames = stack.shape[0]
        field_rotation = self._get_field_rotation(parang=parang)

        # Get the selection_mask for this position, that is, the mask that
        # selects the (spatial) pixels to be used as predictors
        selection_mask = get_selection_mask(mask_size=self.frame_size,
                                            position=position,
                                            field_rotation=field_rotation,
                                            **self.config['selection_mask'])

        # Initialize result variables and set up debugging callbacks
        predictions = np.full(n_frames, np.nan)
        debugging_callbacks = [callback(n_splits=self.splitter.n_splits,
                                        selection_mask=selection_mask)
                               for callback in self.debugging_callbacks]

        # Loop over all train/apply splits, train models and make predictions
        for split_idx, (train_idx, apply_idx) in \
                enumerate(self.splitter.split(n_frames)):

            # Select predictors and targets for both training and application
            train_predictors = stack[train_idx][:, selection_mask]
            apply_predictors = stack[apply_idx][:, selection_mask]
            train_targets = stack[train_idx, position[0], position[1]].reshape(-1, 1)
            apply_targets = stack[apply_idx, position[0], position[1]].reshape(-1, 1)

            # Instantiate a new model and fit it to the training data
            model = self._get_base_model_instance()
            model.fit(X=train_predictors, y=train_targets.ravel())

            # Apply the learned model to the data that was held out
            tmp_predictions = model.predict(X=apply_predictors)

            # Store the predictions
            predictions[apply_idx] = tmp_predictions

            # Run debugging callbacks to collect their data
            for debugging_callback in debugging_callbacks:
                debugging_callback.collect(model=model,
                                           split_idx=split_idx,
                                           y_true=apply_targets,
                                           y_pred=tmp_predictions)

        # Initialize results with the indispensables (position and predictions)
        results = dict(position=position,
                       predictions=predictions)

        # Loop over debugging callbacks and add their data to the results
        for debugging_callback in debugging_callbacks:
            results[debugging_callback.result_name] = debugging_callback.result

        return results

    def get_signal_estimate(self,
                            parang: np.ndarray,
                            subtract: Optional[str] = None,
                            combine: str = 'mean') -> Optional[np.ndarray]:
        """
        Compute the estimate for the planet signal from the residuals.

        Args:
            parang: A numpy array of shape (n_frames, ) containing the
                respective parallactic angles for each image.
            subtract: What to subtract from the residuals before
                derotating them (i.e., "mean" or "median").
            combine: How to combine (average) the derotated residual
                frames (i.e., "mean" or "median").

        Returns:
            A 2D numpy array containing the estimate for the planet
            signal as computed from the residuals.
        """

        # Without residuals, there is no signal estimate
        if self.residuals is None:
            return None

        # Derotate and merge residuals to compute signal estimate
        signal_estimate = derotate_combine(stack=self.residuals,
                                           parang=parang,
                                           mask=(~self.roi_mask),
                                           subtract=subtract,
                                           combine=combine)

        return signal_estimate

    def save_predictions(self) -> None:
        """
        Save predictions as a FITS file to the results directory.
        """

        print(f'Saving predictions to FITS...', end=' ', flush=True)
        self._ensure_results_dir()
        file_path = os.path.join(self.results_dir, 'predictions.fits')
        save_fits(array=self.predictions, file_path=file_path)
        print('Done!', flush=True)

    def save_residuals(self) -> None:
        """
        Save residuals as a FITS file to the results directory.
        """

        print(f'Saving residuals to FITS...', end=' ', flush=True)
        self._ensure_results_dir()
        file_path = os.path.join(self.results_dir, 'residuals.fits')
        save_fits(array=self.residuals, file_path=file_path)
        print('Done!', flush=True)

    def save_debugging_results(self) -> None:
        """
        Save results of debugging callbacks to FITS files.
        """

        # Loop over all debugging results and save them as FITS files
        for key, value in self.debugging_results.items():

            print(f'Saving {key} to FITS...', end=' ', flush=True)
            self._ensure_results_dir()
            file_path = os.path.join(self.results_dir, f'{key}.fits')
            save_fits(array=value, file_path=file_path)
            print('Done!', flush=True)
