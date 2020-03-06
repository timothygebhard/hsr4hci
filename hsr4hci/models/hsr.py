"""
Provides the basic HalfSiblingRegression class.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, NoReturn, Optional, Tuple, Union

import os

from astropy import units
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
from tqdm import tqdm

import numpy as np

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
                 results_dir: str,
                 train_mask: Optional[np.ndarray] = None,
                 store_coefficients: bool = False,
                 store_predictions: bool = False,
                 store_r_squared: bool = False):

        # Store the constructor arguments
        self.config = config
        self.results_dir = results_dir
        self.store_coefficients = store_coefficients
        self.store_predictions = store_predictions
        self.store_r_squared = store_r_squared

        # Define a few useful shortcuts to the configuration
        self.config_model = config['base_model']
        self.experiment_dir = config['experiment_dir']
        self.frame_size = config['dataset']['frame_size']

        # Compute implicitly defined class variables (i.e., the ROI mask)
        self.roi_mask = \
            get_roi_mask(mask_size=self.frame_size,
                         inner_radius=config['roi_mask']['inner_radius'],
                         outer_radius=config['roi_mask']['outer_radius'])

        # Compute the train mask (which is either the ROI itself, or a subset)
        self.train_mask = self.roi_mask
        if train_mask is not None:
            self.train_mask *= train_mask

        # Initialize additional *required* class variables
        self.field_rotation = None
        self.residuals = None

        # Initialize additional *optional* class variables
        if self.store_coefficients:
            self.coefficients = None
        if self.store_predictions:
            self.predictions = None
        if self.store_r_squared:
            self.r_squared = None

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

    def train(self,
              stack: np.ndarray,
              parang: np.ndarray) -> NoReturn:
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

        # Initialize result variables
        self.predictions = np.full(stack.shape, np.nan)
        self.r_squared = np.full((n_splits, *self.frame_size), np.nan)
        self.residuals = np.full(stack.shape, np.nan)
        self.coefficients = \
            np.full((n_splits, *self.frame_size, *self.frame_size), np.nan)

        # Get positions in ROI as a list
        roi_positions = get_positions_from_mask(self.train_mask)

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

        # Once all positions in the ROI have been processed (either in
        # parallel or sequentially), combine all these individual pixel
        # results into a single "global" result
        for result in results:

            # Get the position of the current sub-result
            pos = result['position']

            # Store residuals
            self.residuals[:, pos[0], pos[1]] = result['residuals']

            # Save additional optional results
            if hasattr(self, 'predictions'):
                self.predictions[:, pos[0], pos[1]] = result['predictions']
            if hasattr(self, 'r_squared'):
                self.r_squared[:, pos[0], pos[1]] = result['r_squared']
            if hasattr(self, 'coefficients'):
                self.coefficients[:, pos[0], pos[1]] = result['coefficients']

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

        # Define shortcuts for number of frames and splits
        n_frames = stack.shape[0]
        n_splits = self.config['train_test_splitter']['n_splits']

        # If we have not computed the field rotation of the data set we are
        # using for training, do it now and store the result
        if self.field_rotation is None:
            self.field_rotation = np.abs(parang[-1] - parang[0])
            self.field_rotation = units.Quantity(self.field_rotation, 'degree')

        # Get the selection_mask for this position, that is, the mask that
        # selects the (spatial) pixels to be used as predictors
        selection_mask = get_selection_mask(mask_size=self.frame_size,
                                            position=position,
                                            field_rotation=self.field_rotation,
                                            **self.config['selection_mask'])

        # Create a function to split the stack into training and test
        splitter = TrainTestSplitter(**self.config['train_test_splitter'])

        # Initialize result variables
        predictions = np.full(n_frames, np.nan)
        residuals = np.full(n_frames, np.nan)
        r_squared = np.full(n_splits, np.nan)
        coefficients = np.full((n_splits, *self.frame_size), np.nan)

        # Loop over all train/test splits, train models and make predictions
        for split_idx, (train_idx, test_idx) in \
                enumerate(splitter.split(n_frames)):

            # Select training predictors and targets
            train_predictors = stack[train_idx][:, selection_mask]
            train_targets = stack[train_idx, position[0], position[1]]

            # Instantiate a new model to be trained on these data
            model = self._get_base_model_instance()

            # Fit the model to the training data
            model.fit(X=train_predictors, y=train_targets)

            # Store the coefficients of the model
            if hasattr(model, 'coef_'):
                coefficients[split_idx][selection_mask] = model.coef_

            # Apply the learned model to the test data to get predictions
            test_predictors = stack[test_idx][:, selection_mask]
            tmp_predictions = model.predict(X=test_predictors)

            # Store the predictions
            predictions[test_idx] = tmp_predictions

            # Compute the R^2 value (i.e., fit quality) and store it
            y_true = stack[test_idx, position[0], position[1]]
            r_squared[split_idx] = r2_score(y_true, tmp_predictions)

            # Compute and store the residuals
            residuals[test_idx] = \
                stack[test_idx, position[0], position[1]] - tmp_predictions

        return dict(position=position,
                    predictions=predictions,
                    r_squared=r_squared,
                    residuals=residuals,
                    coefficients=coefficients)

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

    def save_predictions(self):
        """
        Save predictions as a FITS file to the results directory.
        """

        if self.store_predictions:
            print(f'Saving predictions to FITS...', end=' ', flush=True)
            self._ensure_results_dir()
            file_path = os.path.join(self.results_dir, 'predictions.fits')
            save_fits(array=self.predictions, file_path=file_path)
            print('Done!', flush=True)
        else:
            print('No predictions to save!')

    def save_r_squared(self):
        """
        Save R^2 values as a FITS file to the results directory.
        """

        if self.store_r_squared:
            print(f'Saving R^2 values to FITS...', end=' ', flush=True)
            self._ensure_results_dir()
            file_path = os.path.join(self.results_dir, 'r_squared.fits')
            save_fits(array=self.r_squared, file_path=file_path)
            print('Done!', flush=True)
        else:
            print('No r_squared values to save!')

    def save_residuals(self):
        """
        Save residuals as a FITS file to the results directory.
        """

        print(f'Saving residuals to FITS...', end=' ', flush=True)
        self._ensure_results_dir()
        file_path = os.path.join(self.results_dir, 'residuals.fits')
        save_fits(array=self.residuals, file_path=file_path)
        print('Done!', flush=True)

    def save_coefficients(self):
        """
        Save coefficients as a FITS file to the results directory.
        """

        if self.store_coefficients:
            print(f'Saving coefficients to FITS...', end=' ', flush=True)
            self._ensure_results_dir()
            file_path = os.path.join(self.results_dir, 'coefficients.fits')
            save_fits(array=self.coefficients, file_path=file_path)
            print('Done!', flush=True)
        else:
            print('No coefficients to save!')
