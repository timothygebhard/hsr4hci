"""
Prototype class for models (e.g., Half-Sibling Regression).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from abc import abstractmethod, ABC
from typing import Optional

import numpy as np


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class ModelPrototype(ABC):
    """
    Abstract base class for models.
    """

    @abstractmethod
    def train(self,
              stack: np.ndarray,
              parang: Optional[np.ndarray],
              psf_template: Optional[np.ndarray]):
        """
        Abstract training method.

        Args:
            stack: A 3D numpy array containing the training stack.
            parang: A 1D numpy array containing the parallactic angles.
            psf_template: A 2D numpy array containing the PSF template.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self):
        """
        Abstract load method.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self):
        """
        Abstract save method.
        """
        raise NotImplementedError
