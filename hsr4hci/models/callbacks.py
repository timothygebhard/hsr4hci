"""
Classes and methods that can be used as callbacks for models, either
for debugging or to retrieve additional model-specific information.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Tuple

import sys

if sys.version_info < (3, 8):
    from typing_extensions import Protocol
else:
    from typing import Protocol

from sklearn.metrics import r2_score

import numpy as np

from hsr4hci.utils.typehinting import BaseLinearModel, BaseLinearModelCV


# -----------------------------------------------------------------------------
# TYPE DEFINITIONS
# -----------------------------------------------------------------------------

class BaseCollector(Protocol):
    """
    This defines the basic structure that all Collector callbacks should
    follow to ensure they can be used in combination with the HSR class.
    """

    name: str

    def __init__(self, **kwargs: Any): ...

    @classmethod
    def shape(cls,
              frame_size: Tuple[int, int],
              n_splits: int) -> Tuple[int, ...]: ...

    def collect(self,
                split_idx: int,
                **kwargs: Any) -> None: ...

    def get_results(self) -> np.ndarray: ...


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class AlphaCollector(BaseCollector):
    """
    A class that can be used to collect the .alpha_ attribute (i.e.,
    the regularization parameter value) of a model.
    """

    name = 'alphas'

    def __init__(self, **kwargs: Any):

        super().__init__()

        # Store constructor arguments
        self.n_splits = int(kwargs['n_splits'])

        # Initialize the variable in which we keep track of all the alphas
        self.alphas = np.full(self.n_splits, np.nan)

    @classmethod
    def shape(cls,
              frame_size: Tuple[int, int],
              n_splits: int) -> Tuple[int, int, int]:
        return (n_splits,) + frame_size

    def collect(self,
                split_idx: int,
                **kwargs: Any) -> None:

        # Unpack keyword arguments
        model: BaseLinearModelCV = kwargs['model']

        # Make sure the model has an alpha_ attribute
        if hasattr(model, 'alpha_'):
            self.alphas[split_idx] = model.alpha_
        else:
            raise AttributeError('Model has no parameter "alpha_"!')

    def get_results(self) -> np.ndarray:
        return self.alphas


class CoefficientCollector(BaseCollector):
    """
    A class that can be used to collect the .coef_ attributes (i.e.,
    the weights of a linear model) of a model.
    """

    name = 'coefs'

    def __init__(self, **kwargs: Any):

        super().__init__()

        # Store constructor arguments
        self.n_splits = int(kwargs['n_splits'])
        self.selection_mask = np.array(kwargs['selection_mask'])

        # Initialize the variable in which we keep track of the coefficients
        coefs_shape = (self.n_splits, ) + self.selection_mask.shape
        self.coefs = np.full(coefs_shape, np.nan)

    @classmethod
    def shape(cls,
              frame_size: Tuple[int, int],
              n_splits: int) -> Tuple[int, int, int, int, int]:
        return (n_splits,) + frame_size + frame_size

    def collect(self,
                split_idx: int,
                **kwargs: Any) -> None:

        # Unpack keyword arguments
        model: BaseLinearModel = kwargs['model']

        # Make sure the model has a coef_ attribute
        if hasattr(model, 'coef_'):

            # In case we have augmented the predictors beyond the pure pixels,
            # we want to drop these other predictors here
            n_predictor_pixels = int(np.sum(self.selection_mask))
            coefficients = model.coef_[:n_predictor_pixels]

            # Save the coefficients for this split index
            self.coefs[split_idx][self.selection_mask] = coefficients

        else:
            raise AttributeError('Model has no attribute coef_!')

    def get_result(self) -> np.ndarray:
        return self.coefs


class RSquaredCollector(BaseCollector):

    name = 'r_squared'

    def __init__(self, **kwargs: Any):

        super().__init__()

        # Store constructor arguments
        self.n_splits = int(kwargs['n_splits'])

        # Initialize the variable in which we keep track of the R^2 values
        self.r_squared = np.full(self.n_splits, np.nan)

    @classmethod
    def shape(cls,
              frame_size: Tuple[int, int],
              n_splits: int) -> Tuple[int, int, int]:
        return (n_splits, ) + frame_size

    def collect(self,
                split_idx: int,
                **kwargs: Any) -> None:

        # Unpack keyword arguments
        y_true: np.ndarray = np.array(kwargs['y_true'])
        y_pred: np.ndarray = np.array(kwargs['y_pred'])

        # Compute and store the R^2 value of the current split
        self.r_squared[split_idx] = r2_score(y_true=y_true,
                                             y_pred=y_pred)

    def get_results(self) -> np.ndarray:
        return self.r_squared
