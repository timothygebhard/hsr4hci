"""
Additional custom types that can be used for type hinting.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import sys

if sys.version_info < (3, 8):
    from typing_extensions import Protocol
else:
    from typing import Protocol

import numpy as np


# -----------------------------------------------------------------------------
# TYPE DEFINITIONS
# -----------------------------------------------------------------------------

class RegressorModel(Protocol):
    """
    Define a type hint for a generic regressor, that is, a class that
    follows the usual sklearn syntax (i.e., it provides a fit() and a
    predict() method) and can be used to learn a mapping from predictors
    X to targets y.
    """
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RegressorModel': ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


class BaseLinearModel(RegressorModel):
    """
    Define a base class for linear models from sklearn. Linear models
    are characterized by the fact that they have a coefficient vector
    coef_ and an intercept term intercept_.
    """
    coef_: np.ndarray
    intercept_: float


class BaseLinearModelCV(BaseLinearModel):
    """
    Define a base class for cross-validated linear models from sklearn
    such as, e.g., RidgeCV. These models are characterized by the fact
    that they have an alpha_ attribute which stores the value of the
    regularization parameter chosen by the cross-validation.
    """
    alpha_: np.ndarray


class Scaler(Protocol):
    """
    Define type hint for pre-processing scalers (RobustScaler, ...).
    """
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray: ...
    def transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray: ...
    def inverse_transform(self, X: np.ndarray) -> np.ndarray: ...
