"""
Utility functions for preprocessing data.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Optional, Tuple

from scipy.stats import median_absolute_deviation as mad
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, \
    RobustScaler, StandardScaler

import numpy as np

from hsr4hci.utils.typehinting import Scaler


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class NoneScaler(Scaler):
    """
    This is a dummy scaler, which does not actually scale the data at
    all but is useful if we always want to use the same interface.
    """

    def __init__(self) -> None:
        pass

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> None:
        pass

    def transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return X

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return X

    def inverse_transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        return X


class MadScaler(Scaler):
    """
    Scaler that scales the data based on the median, and the median
    absolute deviation (MAD); this should be robust towards outliers.
    """

    def __init__(self) -> None:
        self.median: Optional[np.ndarray] = None
        self.mad: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> None:

        self.median = np.median(X, axis=0)
        self.mad = mad(X, axis=0)

    def transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        if (self.median is not None) and (self.mad is not None):
            return (X - self.median) / self.mad
        raise RuntimeError('Call fit() before you call transform()!')

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.fit(X=X, y=y)
        return self.transform(X=X, y=y)

    def inverse_transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        if (self.median is not None) and (self.mad is not None):
            return X * self.mad + self.median
        raise RuntimeError('Call fit() before you call transform()!')


class PredictorsTargetsScaler:
    """
    A unified interface for accessing different types of scalers to
    scale both the predictors and the targets of a model.

    This class provides simplified access to the following types of
    scalers: None (no scaling), MadScaler, MaxAbsScaler, MinMaxScaler,
    RobustScaler, and StandardScaler.

    Args:
        scaler_type: A string containing the name of the type of scaler
            to be used. Use ``None`` to select no scaling.
    """

    def __init__(
        self,
        scaler_type: Optional[str] = None,
    ) -> None:

        # Store constructor arguments
        self.scaler_type = scaler_type

        # Initialize scalers as NoneScalers to make sure they exist
        self.predictors_scaler: Scaler = NoneScaler()
        self.targets_scaler: Scaler = NoneScaler()

        # Depending on the type of scaler, set up the correct type of
        # scaler for both the predictors and the targets
        if scaler_type is None:
            pass
        elif scaler_type == 'MadScaler':
            self.predictors_scaler = MadScaler()
            self.targets_scaler = MadScaler()
        elif scaler_type == 'MaxAbsScaler':
            self.predictors_scaler = MaxAbsScaler()
            self.targets_scaler = MaxAbsScaler()
        elif scaler_type == 'MinMaxScaler':
            self.predictors_scaler = MinMaxScaler()
            self.targets_scaler = MinMaxScaler()
        elif scaler_type == 'RobustScaler':
            self.predictors_scaler = RobustScaler()
            self.targets_scaler = RobustScaler()
        elif scaler_type == 'StandardScaler':
            self.predictors_scaler = StandardScaler()
            self.targets_scaler = StandardScaler()
        else:
            raise ValueError('Invalid value for "scaler_type"!')

    @staticmethod
    def _fit_transform(
        scaler: Scaler,
        X_train: np.ndarray,
        X_apply: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a given scaler to `X_train`, then apply it to both `X_train`
        and `X_apply`, and return the transformed results.

        Args:
            scaler: A scaler; this should always be one of the two:
                ``self.predictors_scaler`` or ``self.targets_scaler``.
            X_train: Train split of the data.
            X_apply: Apply split of the data (hold out set).

        Returns:
            A tuple ``(X_train_transformed, X_apply_transformed)``.
        """

        # Fit the scaler based on the training data
        scaler.fit(X=X_train, y=None)

        # Scale both the train and apply data
        x_train_scaled = scaler.transform(X=X_train, y=None)
        x_apply_scaled = scaler.transform(X=X_apply, y=None)

        # Return scaled data
        return x_train_scaled, x_apply_scaled

    def fit_transform_predictors(
        self,
        X_train: np.ndarray,
        X_apply: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Call ``_fit_transform()`` on the predictors.
        """
        return self._fit_transform(scaler=self.predictors_scaler,
                                   X_train=X_train,
                                   X_apply=X_apply)

    def fit_transform_targets(
        self,
        X_train: np.ndarray,
        X_apply: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Call ``_fit_transform()`` on the targets.
        """
        return self._fit_transform(scaler=self.targets_scaler,
                                   X_train=X_train.reshape(-1, 1),
                                   X_apply=X_apply.reshape(-1, 1))

    def inverse_transform_predictors(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Call ``inverse_transform()`` for ``self.predictors_scaler``.
        """
        return self.predictors_scaler.inverse_transform(X=X)

    def inverse_transform_targets(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Call ``inverse_transform()`` for ``self.targets_scaler``.
        """
        result = self.targets_scaler.inverse_transform(X=X.reshape(-1, 1))
        return result.ravel()
