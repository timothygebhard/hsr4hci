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
from hsr4hci.utils.predictor_selection import get_predictor_mask
from hsr4hci.utils.roi_selection import get_roi_pixels

from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from tqdm import tqdm
from typing import Tuple
from pathlib import Path


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
        predictor = PixelPredictor(position=position)
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
            predictor = PixelPredictor(position=position)
            predictor.load(models_dir=self.m__models_dir)
            self.m__predictors[position] = predictor

    def save(self):

        # Save all predictors
        for _, predictor in self.m__predictors.items():
            predictor.save(models_dir=self.m__models_dir)

    def get_pca_of_parameters(self,
                              num_components: int=3,
                              normalize: bool=False)-> np.ndarray:

        if not bool(self.m__predictors):
            raise ValueError("No local predictors found. "
                             "Please load models first")

        # Create PCA model
        model_paras = {}
        paras_list = []
        for position, model in self.m__predictors.items():
            # TODO change Ridge to predictor class later
            if isinstance(model.m__model, Ridge):
                model_paras[position] = model.m__model.coef_
                paras_list.append(model.m__model.coef_)

        pca_model = PCA(num_components)
        pca_model.fit(np.array(paras_list))

        # Create result frame and transform coeff
        result = np.zeros((num_components,
                                 self.m__mask_size[0],
                                 self.m__mask_size[1]))

        for position, coeff in model_paras.items():
            result[:, position[0], position[1]] = \
                pca_model.transform(np.array(coeff).reshape(-1, 1))[0, :]

        # Normalize to range [0, 1] to be used as colors
        if normalize:
            # convert to colors
            result = result - np.min(result)
            result /= np.max(result)

        return result


class PixelPredictor(object):
    """
    Wrapper class for a predictor model of a single pixel.
    """

    def __init__(self,
                 position: tuple):

        self.m__position = position
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

        # Instantiate a ridge regression model
        self.m__model = Ridge(alpha=0)

        # Fit to the training data
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
