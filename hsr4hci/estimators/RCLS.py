"""
Regularized Constrained Least Squares using CVXPY.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Tuple, Sequence, Optional

from sklearn.base import RegressorMixin
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.linear_model.base import LinearModel
from sklearn.utils import check_X_y
from sklearn.utils.estimator_checks import check_estimator

import cvxpy as cp
import numpy as np


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

# noinspection PyPep8Naming, PyAttributeOutsideInit, PyUnresolvedReferences
# pylint: disable=attribute-defined-outside-init
class RCLS(LinearModel, RegressorMixin):
    """
    Regularized Constrained Least Squares Regression.

    Provide an estimator that can fit a regularized least squares
    regression (i.e., ridge regression, LASSO, or no regularization)
    while also allowing to enforce constraints on the coefficients of
    the model (e.g., some coefficients must be non-negative)

    Args:
        fit_intercept: Whether or not to fit an intercept term (i.e.,
            a constant offset) in the linear model.
        regularization: The kind of regularization to use for the model.
            Options are None (i.e., no regularization; corresponds to
            regular linear regression), "ridge" (= L2 penalization of
            the coefficients), or "lasso" (= L1 penalization of the
            coefficients).
        alpha: Regularization strength, this must be a positive float.
            This parameter has the same interpretation as the parameter
            of the same name in, e.g., sklearn.linear_model.Ridge.
            If "regularization" is None, this parameter is ignored.

    Attributes:
        coef_ (np.ndarray): A numpy array of shape (n_samples, ) that
            contains the estimated coefficients of the linear model.
        intercept_ (float): Constant offset term in the linear model.
    """

    def __init__(self,
                 fit_intercept: bool = True,
                 regularization: Optional[str] = None,
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5):

        # Store constructor arguments
        self.fit_intercept = fit_intercept

        # Sanity
        if alpha >= 0:
            self.alpha = alpha
        else:
            raise ValueError('alpha must be non-negative!')

        if 0 <= l1_ratio <= 1:
            self.l1_ratio = l1_ratio
        else:
            raise ValueError('l1_ratio must be in [0, 1]!')

        regularization_options = ('ridge', 'lasso', 'elasticnet')
        if regularization is None or regularization in regularization_options:
            self.regularization = regularization
        else:
            raise ValueError('regularization must be one of the following:'
                             'None, "ridge", "lasso", or "elasticnet"!')

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

        coef_bounds = (n_features * [None], n_features * [None])
        intercept_bounds = (None, None)

        return coef_bounds, intercept_bounds

    def _loss_fn(self, X, y):
        return cp.sum_squares(X * self.coef_ - y + self.intercept_)

    def _regularization_term(self):

        if self.regularization is None:
            return 0
        if self.regularization == 'ridge':
            return cp.pnorm(self.coef_, p=2) ** 2
        if self.regularization == 'lasso':
            return cp.pnorm(self.coef_, p=1)
        if self.regularization == 'elasticnet':
            return (self.l1_ratio * cp.pnorm(self.coef_, p=1) +
                    (1 - self.l1_ratio) * 0.5 * cp.pnorm(self.coef_, p=2) ** 2)

        raise ValueError('regularization must be one of the following:'
                         'None, "ridge", "lasso" or "elasticnet"!')

    def _objective_fn(self, X, y):
        return (self._prefactor * self._loss_fn(X, y) +
                self.alpha * self._regularization_term())

    # pylint: disable=arguments-differ
    # TODO: Maybe the bounds should be moved into the constructor?
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            coef_bounds: Optional[Tuple[Sequence, Sequence]] = None,
            intercept_bounds: Optional[Tuple[float, float]] = None):
        """
        Fit a regularized, constrained linear model to the given data.

        Args:
            X: A 2D numpy array of shape (n_samples, n_features).
            y: A 1D numpy array of shape (n_samples, ).
            coef_bounds: The bounds / constraints to be enforced on the
                model's coefficients. This needs to be a tuple of two
                1D arrays, the first of which specifies the lower bound
                on each coefficient, the second one the upper bound.
                Thus, both arrays need to be of shape (n_samples, ).
                By default, all the bounds for all coefficients are
                None, i.e, they are effectively unconstrained.
            intercept_bounds: The bounds / constraints to be enforced
                on the model's intercept. This needs to be a tuple of
                floats. By default, (None, None) is used, that is, the
                intercept is effectively unconstrained.

        Returns:
            self: Returns an instance of self.
        """

        # Ensure X and y have the correct shape for a sklearn-style estimator
        X, y = check_X_y(X, y, y_numeric=True)
        n_samples, n_features = X.shape

        # Compute the pre-factor for the objective function (this is necessary
        # to make the results consistent with the sklearn)
        if self.regularization in ('lasso', 'elasticnet'):
            self._prefactor = 1 / (2 * n_samples)
        else:
            self._prefactor = 1

        # Construct the coefficients and the intercept as CVXPY variables
        self.coef_ = cp.Variable(n_features)
        if self.fit_intercept:
            self.intercept_ = cp.Variable(1)
        else:
            self.intercept_ = cp.Constant(0)

        # Get default bounds for the coefficients and the intercept
        default_coef_bounds, default_intercept_bounds = \
            self._get_default_bounds(n_features=n_features)
        if coef_bounds is None:
            coef_bounds = default_coef_bounds
        if intercept_bounds is None:
            intercept_bounds = default_intercept_bounds

        # Initialize list of constraints of coefficients and intercept
        constraints = list()

        # Take care of bounds on the coefficients
        for i in range(n_features):
            if coef_bounds[0][i] is not None:
                constraints += [self.coef_[i] >= coef_bounds[0][i]]
            if coef_bounds[1][i] is not None:
                constraints += [self.coef_[i] <= coef_bounds[1][i]]

        # Take care of bounds on the intercept:
        if intercept_bounds[0] is not None:
            constraints += [self.intercept_ >= intercept_bounds[0]]
        if intercept_bounds[1] is not None:
            constraints += [self.intercept_ <= intercept_bounds[1]]

        # Define the optimization problem and (try to) solve it
        objective = cp.Minimize(self._objective_fn(X=X, y=y))
        problem = cp.Problem(objective=objective, constraints=constraints)
        problem.solve(solver='SCS')

        # TODO: This solution is still unsatisfying!
        # If the solver failed, we can try again without the constraints?
        if self.intercept_.value is None:
            print('WARNING: Optimization failed, trying again without '
                  'constraints!')
            problem = cp.Problem(objective=objective)
            problem.solve(solver='SCS')

        # Cast the results to a numpy array / a float
        self.coef_ = np.array(self.coef_.value)
        self.intercept_ = float(self.intercept_.value)

        return self


class ExoplanetRCLS(RCLS):
    """
    RCLS for forward modeling-based planet search.
    """

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
            default values for the bounds, which are None for all
            coefficients except for the last one, which is lower-
            bounded by 0.
        """

        coef_bounds = ((n_features - 1) * [None] + [0], n_features * [None])
        intercept_bounds = (None, None)

        return coef_bounds, intercept_bounds


# -----------------------------------------------------------------------------
# TEST AREA
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Fix random seed
    np.random.seed(42)

    # Make sure the estimator complies with the sklearn API for estimators
    check_estimator(RCLS)
    print()

    # Construct a dummy dataset to compare our estimator with sklearn
    X, y = make_regression(n_samples=1000, n_features=4,
                           n_informative=2, bias=1.0)

    options = dict(alpha=10)

    rcls = RCLS(regularization='ridge', **options)
    rcls.fit(X=X, y=y)

    skl = Ridge(**options)
    skl.fit(X=X, y=y)

    with np.printoptions(precision=6, suppress=True):
        print('Coefficients:')
        print('RCLS:   ', rcls.coef_)
        print('sklearn:', skl.coef_)
        print()
        print('Intercept:')
        print('RCLS:   ', rcls.intercept_)
        print('sklearn:', skl.intercept_)
        print()
