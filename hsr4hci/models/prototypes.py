"""
Prototype class for models (which are used for noise estimation; e.g. HSR).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

from abc import abstractmethod, ABC
from typing import Optional


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class ModelPrototype(ABC):

    @abstractmethod
    def train(self,
              stack: np.ndarray,
              parang: Optional[np.ndarray],
              psf_template: Optional[np.ndarray]):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError
