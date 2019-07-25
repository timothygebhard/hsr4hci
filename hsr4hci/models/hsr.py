"""
Half-Sibling Regression model.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import joblib
import numpy as np
import os

from hsr4hci.models.prototypes import ModelPrototype
from hsr4hci.utils.predictor_selection import get_predictor_mask
from hsr4hci.utils.roi_selection import get_roi_pixels

from sklearn.linear_model import Ridge
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
                 experiment_dir: str):

        self.m__experiment_dir = experiment_dir

        # Define a models directory and ensure it exists
        self.m__models_dir = os.path.join(self.m__experiment_dir, 'models')
        Path(self.m__models_dir).mkdir(exist_ok=True)
        
        self.m__predictors = dict()

        # TODO: Read in experiment config

    def train(self,
              training_stack: np.ndarray):

        # Get positions of pixels in ROI
        roi_pixels = get_roi_pixels(mask_size=tuple(training_stack.shape[1:]),
                                    pixscale=0.0271,
                                    inner_exclusion_radius=0.15,
                                    outer_exclusion_radius=0.70)

        # Train a model for every position
        for position in roi_pixels:
            self.train_position(position=position,
                                training_stack=training_stack)

    def train_position(self,
                       position: Tuple[int],
                       training_stack: np.ndarray):

        # Get sources mask
        mask = get_predictor_mask(mask_size=tuple(training_stack.shape[1:]),
                                  position=position,
                                  n_regions=1,
                                  region_size=5)

        # Select sources (predictor pixels) and targets from stack
        sources = training_stack[mask]
        targets = training_stack[position]

        # Train and save a predictor for this position
        predictor = PixelPredictor(position=position)
        predictor.train(sources=sources, targets=targets)
        predictor.save(models_dir=self.m__models_dir)

        # Add to dictionary of trained predictors
        self.m__predictors[position] = predictor

    def predict(self,
                test_stack: np.ndarray):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError


class PixelPredictor(object):

    def __init__(self,
                 position: tuple):

        self.m__position = position
        self.m__name = f'{position[0]}_{position[1]}__model.pkl'
        self.m__model = None

    def train(self,
              sources: np.ndarray,
              targets: np.ndarray):

        # Instantiate a ridge regression model
        self.m__model = Ridge(alpha=0)

        # Fit to the training data
        self.m__model.fit(X=sources, y=targets)

    def save(self,
             models_dir: str):

        file_path = os.path.join(models_dir, self.m__name)
        joblib.dump(self.m__model, filename=file_path)

    def load(self,
             models_dir: str):

        file_path = os.path.join(models_dir, self.m__name)
        self.m__model = joblib.load(filename=file_path)
