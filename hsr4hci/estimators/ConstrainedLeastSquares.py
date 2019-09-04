"""
A simple ConstrainedLeastSquares estimator.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

from scipy.optimize import lsq_linear
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.utils import check_X_y
from sklearn.utils.estimator_checks import check_estimator
from typing import Tuple, Sequence, Optional


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

# noinspection PyPep8Naming, PyAttributeOutsideInit, PyUnresolvedReferences
class ConstrainedLeastSquares(LinearModel, RegressorMixin):
    """
    Least Squares Regression with constraints on the coefficients.
    
    Args:
        fit_intercept: Whether or not to fit an intercept term (i.e.,
            a constant offset) in the linear model.

    Attributes:
        coef_ (np.ndarray): A numpy array of shape (n_samples, ) that
            contains the estimated coefficients of the linear model.
        intercept_ (float): Constant offset term in the linear model.
    """

    def __init__(self,
                 fit_intercept: bool = True):
        self.fit_intercept = fit_intercept

    @staticmethod
    def _get_default_bounds(n_features: int) -> Tuple[tuple, tuple]:
        """
        Get the default bounds for the coefficients and the intercept.

        Args:
            n_features: The number of features of the data X, that is,
                the number of coefficients in the model (excluding the
                intercept term).

        Returns:
            A tuple `(coef_bounds, intercept_bounds)`, containing the
            default values for the bounds (which correspond to an
            unconstrained linear model).
        """

        coef_bounds = (np.full(n_features, -np.inf),
                       np.full(n_features, np.inf))
        intercept_bounds = (-np.inf, np.inf)

        return coef_bounds, intercept_bounds

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            coef_bounds: Optional[Tuple[Sequence, Sequence]] = None,
            intercept_bounds: Optional[Tuple[float, float]] = None,
            **kwargs):
        """
        Fit a constrained linear model to the given data.

        Args:
            X: A 2D numpy array of shape (n_samples, n_features).
            y: A 1D numpy array of shape (n_samples, ).
            coef_bounds: The bounds / constraints to be enforced on the
                model's coefficients. This needs to be a tuple of two
                1D arrays, the first of which specifies the lower bound
                on each coefficient, the second one the upper bound.
                Thus, both arrays need to be of shape (n_samples, ).
                By default, all coefficients are constrained to be in
                (-inf, inf), i.e, they are effectively unconstrained.
            intercept_bounds: The bounds / constraints to be enforced
                on the model's intercept. This needs to be a tuple of
                floats. By default, (-inf, inf) is used, that is, the
                intercept is effectively unconstrained.
            **kwargs: More keyword arguments that are passed directly
                to the `scipy.optimize.lsq_linear` method that is used
                to fit the coefficients and the intercept of the model.

        Returns:
            self: Returns an instance of self.
        """

        # Ensure X and y have the correct shape for a sklearn-style estimator
        X, y = check_X_y(X, y, y_numeric=True)

        # Get default bounds for coefficients and intercept
        default_coef_bounds, default_intercept_bounds = \
            self._get_default_bounds(n_features=X.shape[1])
        if coef_bounds is None:
            coef_bounds = default_coef_bounds
        if intercept_bounds is None:
            intercept_bounds = default_intercept_bounds

        # If necessary, add constant a column to the data to fit an intercept.
        # Furthermore, construct the full bounds by combining the bounds for
        # the coefficients and the intercept.
        if self.fit_intercept:
            X = np.column_stack([X, np.ones(X.shape[0])])
            bounds = \
                tuple([np.array(list(coef_bounds[0])+[intercept_bounds[0]]),
                       np.array(list(coef_bounds[1])+[intercept_bounds[1]])])
        else:
            bounds = coef_bounds

        # Use scipy.optimize.lsq_linear to fit the constrained linear model
        # and get the coefficients (this does all the hard work!)
        coefficients = lsq_linear(X, y, bounds=bounds, **kwargs).x

        # Separate the intercept and the coefficients and store the results
        if self.fit_intercept:
            self.coef_ = coefficients[:-1]
            self.intercept_ = coefficients[-1]
        else:
            self.coef_ = coefficients
            self.intercept_ = 0

        return self


# -----------------------------------------------------------------------------
# TEST AREA
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Make sure the estimator complies with the sklearn API for estimators
    check_estimator(ConstrainedLeastSquares)
