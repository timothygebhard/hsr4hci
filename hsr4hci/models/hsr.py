"""
Half-Sibling Regression model.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

from hsr4hci.models.prototypes import ModelPrototype

from typing import Tuple


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
        self.m__predictors = dict()

        # TODO: Read in experiment config

    def train(self,
              training_stack: np.ndarray):
        raise NotImplementedError

    def train_position(self,
                       position: Tuple[int],
                       training_stack: np.ndarray):
        raise NotImplementedError

    def predict(self,
                test_stack: np.ndarray):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError
