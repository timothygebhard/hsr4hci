"""
Half-Sibling Regression model.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import joblib
import numpy as np
import os
import warnings

from hsr4hci.models.prototypes import ModelPrototype
from hsr4hci.utils.model_loading import get_class_by_name
from hsr4hci.utils.predictor_selection import get_predictor_mask
from hsr4hci.utils.roi_selection import get_roi_pixels

from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm
from typing import Tuple


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class HalfSiblingRegression(ModelPrototype):
    """
    Wrapper class for a half-sibling regression model.
    """

    def __init__(self,
                 config: dict):

        # Store the experiment configuration
        self.m__pixscale = config['dataset']['pixscale']
        self.m__roi_ier = config['experiment']['roi']['inner_exclusion_radius']
        self.m__roi_oer = config['experiment']['roi']['outer_exclusion_radius']
        self.m__model_config = config['experiment']['model']

        # Define shortcuts to config elements
        self.m__experiment_dir = config['experiment_dir']
        self.m__mask_size = (int(config['dataset']['x_size']),
                             int(config['dataset']['y_size']))

        # Define a models directory and ensure it exists
        self.m__models_dir = os.path.join(self.m__experiment_dir, 'models')
        Path(self.m__models_dir).mkdir(exist_ok=True)

        # Initialize a dict that will hold all pixel models
        self.m__predictors = dict()

    def train(self,
              training_stack: np.ndarray):

        # Get positions of pixels in ROI
        roi_pixels = get_roi_pixels(mask_size=self.m__mask_size,
                                    pixscale=self.m__pixscale,
                                    inner_exclusion_radius=self.m__roi_ier,
                                    outer_exclusion_radius=self.m__roi_oer)

        # Train a model for every position
        for position in tqdm(roi_pixels, total=len(roi_pixels), ncols=80):
            self.train_position(position=position,
                                training_stack=training_stack)

    def train_position(self,
                       position: Tuple[int],
                       training_stack: np.ndarray):

        # Get sources mask
        mask = get_predictor_mask(mask_size=self.m__mask_size,
                                  position=position,
                                  n_regions=1,
                                  region_size=5)

        # Select sources (predictor pixels) and targets from stack
        sources = training_stack[:, mask]
        targets = training_stack[:, position[0], position[1]]

        # Train and save a predictor for this position
        predictor = PixelPredictor(position=position,
                                   model_config=self.m__model_config)
        predictor.train(sources=sources, targets=targets)
        predictor.save(models_dir=self.m__models_dir)

        # Add to dictionary of trained predictors
        self.m__predictors[position] = predictor

    def predict(self,
                test_stack: np.ndarray):

        # Instantiate empty array to hold all predictions
        predictions = np.full(test_stack.shape, np.nan)

        # Loop over all ROI positions / models
        for position, predictor in tqdm(self.m__predictors.items(), ncols=80,
                                        total=len(self.m__predictors)):

            # Get sources mask
            mask = get_predictor_mask(mask_size=self.m__mask_size,
                                      position=position,
                                      n_regions=1,
                                      region_size=5)
            sources = test_stack[:, mask]

            # Make prediction
            predictions[:, position[0], position[1]] = \
                predictor.predict(sources=sources)

        return predictions

    def load(self):

        # Get positions of pixels in ROI
        roi_pixels = get_roi_pixels(mask_size=self.m__mask_size,
                                    pixscale=self.m__pixscale,
                                    inner_exclusion_radius=self.m__roi_ier,
                                    outer_exclusion_radius=self.m__roi_oer)

        # Load model for every position in the ROI
        for position in roi_pixels:
            predictor = PixelPredictor(position=position,
                                       model_config=self.m__model_config)
            predictor.load(models_dir=self.m__models_dir)
            self.m__predictors[position] = predictor

    def save(self):

        # Save all predictors
        for _, predictor in self.m__predictors.items():
            predictor.save(models_dir=self.m__models_dir)

    def get_pca_of_coefficients(self,
                                n_components: int = 3,
                                normalize: bool = False) -> np.ndarray:
        """
        Run PCA on the coefficients of the pixel predictors. This
        dimensionality reduction is useful to investigate if predictors
        that are spatially close also have similar components (i.e.,
        they have learned a similar relation between input and output).
        
        Args:
            n_components: Number of principal components to project
                onto. Default is 3, as it allows to interpret the
                result as an RGB color.
            normalize: Whether or not to normalize all values in the
                result into the [0, 1] interval.

        Returns:
            A numpy array of shape (n_components, width, height) that
            contains the dimensionality-reduced model coefficients of
            every pixel predictor.
        """

        # Check if we have some models on which we can run PCA.
        # Empty dictionaries are evaluated as False by Python.
        if not self.m__predictors:
            raise ValueError("No local predictors found. "
                             "Please train or load models first.")

        # Collect coefficients for all pixel models
        model_coeffs = {}
        for position, model in self.m__predictors.items():
            if model.m__model.coef_ is not None:
                model_coeffs[position] = model.m__model.coef_

        # Fit a PCA to the coefficients
        pca_model = PCA(n_components)
        pca_model.fit(np.array(list(model_coeffs.values())))

        # Create result array (default to NaN, which is ignored in plots)
        result = np.full((n_components,
                          self.m__mask_size[0],
                          self.m__mask_size[1]), np.nan)

        # For each pixel model, project the coefficients onto the PCs
        for position, coeff in model_coeffs.items():
            result[:, position[0], position[1]] = \
                pca_model.transform(np.array(coeff).reshape(-1, 1))[0, :]

        # If desired, normalize result to range [0, 1] to be used as colors
        if normalize:
            result = result - np.nanmin(result)
            result /= np.nanmax(result)

        return result


class PixelPredictor(object):
    """
    Wrapper class for a predictor model of a single pixel.
    """

    def __init__(self,
                 position: tuple,
                 model_config: dict):

        # Store constructor arguments
        self.m__position = position
        self.m__model_config = model_config

        # Create predictor name and placeholder for model
        self.m__name = f'{position[0]}_{position[1]}__model.pkl'
        self.m__model = None

    @property
    def coef_(self):

        # If the base model has a coef_ attribute (which only exists for
        # fitted models), we can return it; otherwise return None
        if hasattr(self.m__model, 'coef_'):
            return self.m__model.coef_
        return None

    def train(self,
              sources: np.ndarray,
              targets: np.ndarray):

        # Instantiate a new model according to the model_config
        model_class = \
            get_class_by_name(module_name=self.m__model_config['module'],
                              class_name=self.m__model_config['class'])
        self.m__model = model_class(**self.m__model_config['parameters'])

        # Fit model to the training data
        self.m__model.fit(X=sources, y=targets)

    def predict(self,
                sources: np.ndarray) -> np.ndarray:

        return self.m__model.predict(X=sources)

    def save(self,
             models_dir: str):

        file_path = os.path.join(models_dir, self.m__name)
        joblib.dump(self.m__model, filename=file_path)

    def load(self,
             models_dir: str):

        # Try to load the model for this predictor from its *.pkl file
        file_path = os.path.join(models_dir, self.m__name)
        if os.path.isfile(file_path):
            self.m__model = joblib.load(filename=file_path)
        else:
            warnings.warn(f'Model file not found: {file_path}')
