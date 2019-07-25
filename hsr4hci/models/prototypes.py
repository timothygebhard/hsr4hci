"""
Prototype class for models (which are used for noise estimation; e.g. HSR).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

from abc import abstractmethod, ABC


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class ModelPrototype(ABC):

    @abstractmethod
    def train(self,
              training_stack: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def predict(self,
                test_stack: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError
