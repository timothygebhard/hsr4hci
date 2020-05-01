"""
Classes and methods that can be used as callbacks for models, either
for debugging or to retrieve additional model-specific information.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Tuple, Union

from sklearn.metrics import r2_score

import numpy as np


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class AlphaCollector:
    """
    A class that can be used to collect the .alpha_ attribute (i.e.,
    the regularization parameter value) of a model.
    """

    # Define name that will be used for returning the result.
    # Defining this on the class level means we can access it without having
    # to instantiate the class first.
    result_name = 'alphas'

    def __init__(self,
                 n_splits: int,
                 **_):

        # Store constructor arguments
        self.n_splits = n_splits

        # Initialize the variable in which we keep track of all the alphas
        self.alphas = np.full(n_splits, np.nan)

    @classmethod
    def shape(cls,
              frame_size: Tuple[int, int],
              n_splits: int) -> Tuple[int, int, int]:
        return (n_splits,) + frame_size

    def collect(self,
                model: Any,
                split_idx: int,
                **_) -> None:

        # If the model we have gotten has an alpha parameter, and we have
        # received a valid split index, we can store the alpha value of the
        # current model at the respective position
        if hasattr(model, 'alpha_') and (0 <= split_idx < self.n_splits):
            self.alphas[split_idx] = model.alpha_

    @property
    def result(self) -> np.ndarray:
        return self.alphas


class CoefficientCollector:
    """
    A class that can be used to collect the .coef_ attributes (i.e.,
    the weights of a linear model) of a model.
    """

    # Define name that will be used for returning the result.
    # Defining this on the class level means we can access it without having
    # to instantiate the class first.
    result_name = 'coefs'

    def __init__(self,
                 n_splits: int,
                 selection_mask: np.array,
                 **_):

        # Store constructor arguments
        self.n_splits = n_splits
        self.selection_mask = selection_mask

        # Initialize the variable in which we keep track of the coefficients
        self.coefs = np.full((n_splits,) + selection_mask.shape, np.nan)

    @classmethod
    def shape(cls,
              frame_size: Tuple[int, int],
              n_splits: int) -> Tuple[int, int, int, int, int]:
        return (n_splits,) + frame_size + frame_size

    def collect(self,
                model,
                split_idx: int,
                **_) -> None:

        # If the model we have gotten has an alpha parameter, and we have
        # received a valid split index, we can store the alpha value of the
        # current model at the respective position
        if hasattr(model, 'coef_') and (0 <= split_idx < self.n_splits):

            # In case we have augmented the predictors beyond the pure pixels,
            # we want to drop these other predictors here
            n_predictor_pixels = int(np.sum(self.selection_mask))
            coefficients = model.coef_[:n_predictor_pixels]

            # Save the coefficients for this split index
            self.coefs[split_idx][self.selection_mask] = coefficients

    @property
    def result(self) -> np.ndarray:
        return self.coefs


class RSquaredCollector:

    # Define name that will be used for returning the result.
    # Defining this on the class level means we can access it without having
    # to instantiate the class first.
    result_name = 'r_squared'

    def __init__(self,
                 n_splits: int,
                 **_):

        # Store constructor arguments
        self.n_splits = n_splits

        # Initialize the variable in which we keep track of the R^2 values
        self.r_squared = np.full(n_splits, np.nan)

    @classmethod
    def shape(cls,
              frame_size: Tuple[int, int],
              n_splits: int) -> Tuple[int, int, int]:
        return (n_splits,) + frame_size

    def collect(self,
                split_idx: int,
                y_true: np.ndarray,
                y_pred: np.ndarray,
                **_) -> None:

        # If we have received a valid split index, we can compute and store
        # the R^2 value of the current split at the respective position
        if 0 <= split_idx < self.n_splits:
            self.r_squared[split_idx] = r2_score(y_true=y_true,
                                                 y_pred=y_pred)

    @property
    def result(self) -> np.ndarray:
        return self.r_squared


# -----------------------------------------------------------------------------
# TYPE DEFINITIONS
# -----------------------------------------------------------------------------

DebuggingCollector = Union[AlphaCollector,
                           CoefficientCollector,
                           RSquaredCollector]
