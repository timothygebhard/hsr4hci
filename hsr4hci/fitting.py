"""
Utility functions for fitting (e.g., PSFs with analytical models).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pprint import pformat
from typing import Any, Dict, Tuple

import math
import warnings

from scipy.optimize import curve_fit

import numpy as np


# -----------------------------------------------------------------------------
# BASE CLASS
# -----------------------------------------------------------------------------

class Fittable2DModel:
    """
    This class implements the basic functionality that is shared between
    all the fittable 2D models below.
    """

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        """
        Define parameter names (and parameter order). This has to be
        implemented by every model individually.
        """
        raise NotImplementedError

    @property
    def parameters(self) -> Tuple[float, ...]:
        """
        Get the parameters or the model, in the order determined by the
        `parameter_names` (i.e., the order in which they need to be
        passed to the model constructor).

        Returns:
            A tuple of floats containing the model parameters.
        """
        return tuple(getattr(self, _) for _ in self.parameter_names)

    @property
    def named_parameters(self) -> Dict[str, float]:
        """
        Get a dictionary that contains a mapping between the parameter
        names and the respective parameter values.

        Returns:
            A dictionary that maps parameter names to parameter values.
        """
        return dict(zip(self.parameter_names, self.parameters))

    def set_parameters(self, *parameters: float) -> None:
        """
        Update the member variables that contain the model parameters.

        Args:
            *parameters: A tuple of floats that contain the new values
                for the model parameters (in the correct order).
        """
        for name, value in zip(self.parameter_names, parameters):
            setattr(self, name, value)

    @staticmethod
    def evaluate(
        meshgrid: Tuple[np.ndarray, np.ndarray],
        *parameters: float,
    ) -> np.ndarray:
        """
        This function needs to be implemented by every model. It
        contains the basic functional form of the model, e.g., a 2D
        Gaussian or a 2D Moffat function.

        Note: Because this function is used for fitting the model, it
            does not use the parameter values stored in the class
            members. Instead, you need to pass the parameter values
            explicitly using the `*parameters` argument.

        Args:
            meshgrid: A 2-tuple of numpy arrays containing the meshgrid
                with the coordinates (x- and y-values).
            *parameters: An ordered tuple (the order needs to match the
                `parameter_names`) containing the values to be used for
                the model.

        Returns:
            The value of the model on the given `meshgrid` when using
            the parameter values in `*parameters`.
        """
        raise NotImplementedError

    def evaluate_without_offset(
        self,
        meshgrid: Tuple[np.ndarray, np.ndarray],
        *parameters: float,
    ) -> np.ndarray:
        """
        The same as `evaluate()`, except that the `offset` parameter,
        which by convention always is the last parameter in the order
        defined by the `parameter_names`, is set to 0.

        Args:
            meshgrid: A 2-tuple of numpy arrays containing the meshgrid
                with the coordinates (x- and y-values).
            *parameters: An ordered tuple (the order needs to match the
                `parameter_names`) containing the values to be used for
                the model.

        Returns:
            The value of the model on the given `meshgrid` when using
            the parameter values in `*parameters`.
        """
        return self.evaluate(meshgrid, *parameters[:-1], 0)

    def fit(
        self,
        meshgrid: Tuple[np.ndarray, np.ndarray],
        target: np.ndarray,
        use_offset: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Fit the model as a function that maps the given `meshgrid` onto
        the given `target`.

        Args:
            meshgrid: A 2-tuple of numpy arrays containing the meshgrid
                with the coordinates (x- and y-values).
            target: A 2D numpy array containing the target values for
                the fit (e.g., a PSF template).
            use_offset: Whether or not to fit a constant offset. If this
                is False, the `offset` is forced to 0.
            **kwargs: Additional keyword arguments that will be passed
                to `scipy.optimize.curve_fit()`; for example, this can
                be used to pass a `bounds` argument.
        """

        # Define target function and wrap it so that its outputs are raveled
        def target_function(*args: Any, **kwargs: Any) -> np.ndarray:
            if use_offset:
                return self.evaluate(*args, **kwargs).ravel()
            return self.evaluate_without_offset(*args, **kwargs).ravel()

        # Fit the target using the initial parameter guess
        with warnings.catch_warnings():

            # Ignore some numpy warnings here that can happen if the minimizer
            # explores a particularly bad parameter range
            warnings.filterwarnings('ignore', r'invalid value encountered')
            warnings.filterwarnings('ignore', r'overflow encountered in')
            warnings.filterwarnings('ignore', r'Covariance of the')

            # Actually fit the given target using our target function
            parameters, _ = curve_fit(
                f=target_function,
                xdata=meshgrid,
                ydata=np.nan_to_num(target).ravel(),
                p0=self.parameters,
                **kwargs,
            )

        # Update the model using the best fit parameters
        self.set_parameters(*parameters)

    def __call__(self, meshgrid: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Evaluate the model -- using parameter values that are stored
        in self.parameters -- on the given meshgrid.

        Args:
            meshgrid: A 2-tuple of numpy arrays containing the meshgrid
                with the coordinates (x- and y-values).

        Returns:
            A 2D numpy array containing the values of the model on the
            given meshgrid.
        """
        return self.evaluate(meshgrid, *self.parameters)

    def __repr__(self) -> str:
        """
        The string representation of the model.

        Returns:
            A string containing a human-readable summary of the model
            parameters.
        """
        return pformat(vars(self))


# -----------------------------------------------------------------------------
# MODEL CLASSES
# -----------------------------------------------------------------------------

class CircularGauss2D(Fittable2DModel):
    """
    A circular (i.e., symmetric) 2D Gauss function.

    Args:
        mu_x: The x-position of the center of the Gauss function.
        mu_y: The y-position of the center of the Gauss function.
        sigma: The standard deviation of the Gauss function.
        amplitude: The (peak) amplitude of the Gauss function.
        offset: Global offset (e.g., the background flux level).
    """

    def __init__(
        self,
        mu_x: float = 0.0,
        mu_y: float = 0.0,
        sigma: float = 1.0,
        amplitude: float = 1.0,
        offset: float = 0.0,
    ) -> None:

        self.mu_x = mu_x
        self.mu_y = mu_y
        self.sigma = sigma
        self.amplitude = amplitude
        self.offset = offset

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        """
        Define parameter names (and parameter order).
        """
        return 'mu_x', 'mu_y', 'sigma', 'amplitude', 'offset'

    @property
    def fwhm(self) -> float:
        """
        Compute the full width half maximum (FWHM) for the model.
        """
        return 2 * math.sqrt(2 * math.log(2)) * self.sigma

    @staticmethod
    def evaluate(
        meshgrid: Tuple[np.ndarray, np.ndarray],
        *parameters: float,
    ) -> np.ndarray:

        # Unpack the meshgrid as well as the parameters
        xx_grid, yy_grid = meshgrid
        mu_x, mu_y, sigma, amplitude, offset = parameters

        # Compute (x - mu_x) and (y - mu_y)
        x_diff = xx_grid - mu_x
        y_diff = yy_grid - mu_y

        # Compute exponent of the Gaussian
        inner = (x_diff ** 2 + y_diff ** 2) / (2 * sigma ** 2)

        # Combine all parts into the final function
        return np.asarray(amplitude * np.exp(-inner) + offset)

    def __repr__(self) -> str:
        return pformat({**self.named_parameters, **{'fwhm': self.fwhm}})


class EllipticalGauss2D(Fittable2DModel):
    """
    An elliptical 2D Gauss function.

    Args:
        mu_x: The x-position of the center of the Gauss function.
        mu_y: The y-position of the center of the Gauss function.
        sigma_x: The standard deviation in x-direction.
        sigma_y: The standard deviation in y-direction.
        theta: Rotation angle (in radian), measured counter-clockwise.
        amplitude: The (peak) amplitude of the Gauss function.
        offset: Global offset (e.g., the background flux level).
    """

    def __init__(
        self,
        mu_x: float = 0.0,
        mu_y: float = 0.0,
        sigma_x: float = 1.0,
        sigma_y: float = 1.0,
        theta: float = 0.0,
        amplitude: float = 1.0,
        offset: float = 0.0,
    ) -> None:

        self.mu_x = mu_x
        self.mu_y = mu_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.theta = theta
        self.amplitude = amplitude
        self.offset = offset

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        """
        Define parameter names (and parameter order).
        """
        return (
            'mu_x',
            'mu_y',
            'sigma_x',
            'sigma_y',
            'theta',
            'amplitude',
            'offset',
        )

    @property
    def fwhm_x(self) -> float:
        """
        Compute the full width half maximum (FWHM) in x-direction.
        """
        return 2 * math.sqrt(2 * math.log(2)) * self.sigma_x

    @property
    def fwhm_y(self) -> float:
        """
        Compute the full width half maximum (FWHM) in y-direction.
        """
        return 2 * math.sqrt(2 * math.log(2)) * self.sigma_y

    @staticmethod
    def evaluate(
        meshgrid: Tuple[np.ndarray, np.ndarray],
        *parameters: float,
    ) -> np.ndarray:

        # Unpack the meshgrid as well as the parameters
        xx_grid, yy_grid = meshgrid
        mu_x, mu_y, sigma_x, sigma_y, theta, amplitude, offset = parameters

        # Compute (x - mu_x) and (y - mu_y)
        x_diff = xx_grid - mu_x
        y_diff = yy_grid - mu_y

        # Compute the coefficients A, B and C
        a = (np.cos(theta) ** 2 / (2 * sigma_x ** 2)) + (
            np.sin(theta) ** 2 / (2 * sigma_y ** 2)
        )
        b = (np.sin(2 * theta) / (2 * sigma_x ** 2)) - (
            np.sin(2 * theta) / (2 * sigma_y ** 2)
        )
        c = (np.sin(theta) ** 2 / (2 * sigma_x ** 2)) + (
            np.cos(theta) ** 2 / (2 * sigma_y ** 2)
        )

        # Compute exponent of Gaussian
        inner = (a * x_diff ** 2) + (b * x_diff * y_diff) + (c * y_diff ** 2)

        # Combine all parts into the final function
        return np.asarray(amplitude * np.exp(-inner) + offset)

    def __repr__(self) -> str:
        return pformat(
            {
                **self.named_parameters,
                **{'fwhm_x': self.fwhm_x, 'fwhm_y': self.fwhm_y},
            }
        )


class CircularMoffat2D(Fittable2DModel):
    """
    A circular (i.e., symmetric) 2D Moffat function.

    This function uses a parametrization that is based on the one given
    in the AsPyLib library:
        http://www.aspylib.com/doc/aspylib_fitting.html#circular-moffat-psf

    Args:
        mu_x: The x-position of the center of the Moffat function.
        mu_y: The y-position of the center of the Moffat function.
        alpha: The alpha parameter of the Moffat function.
        amplitude: The (peak) amplitude of the Moffat function.
        offset: Global offset (e.g., the background flux level).
        beta: The beta parameter of the Moffat function. For `beta = 1`,
            the Moffat function is essentially just a Cauchy function
            (also known as Lorentz or Breit-Wigner).
    """

    def __init__(
        self,
        mu_x: float = 0.0,
        mu_y: float = 0.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        amplitude: float = 1.0,
        offset: float = 0.0,
    ) -> None:

        self.mu_x = mu_x
        self.mu_y = mu_y
        self.alpha = alpha
        self.beta = beta
        self.amplitude = amplitude
        self.offset = offset

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        """
        Define parameter names (and parameter order).
        """
        return 'mu_x', 'mu_y', 'alpha', 'beta', 'amplitude', 'offset'

    @property
    def fwhm(self) -> float:
        """
        Compute the full width half maximum (FWHM) for the model.
        """
        return 2 * math.sqrt(2 ** (1 / self.beta) - 1) * self.alpha

    @staticmethod
    def evaluate(
        meshgrid: Tuple[np.ndarray, np.ndarray],
        *parameters: float,
    ) -> np.ndarray:

        # Unpack the meshgrid as well as the parameters
        xx_grid, yy_grid = meshgrid
        mu_x, mu_y, alpha, beta, amplitude, offset = parameters

        # Compute (x - x_0) and (y - y_0)
        x_diff = xx_grid - mu_x
        y_diff = yy_grid - mu_y

        # Combine all parts into the final function
        return np.asarray(
            amplitude
            / (1 + (x_diff ** 2 + y_diff ** 2) / (alpha ** 2)) ** beta
            + offset
        )

    def __repr__(self) -> str:
        return pformat({**self.named_parameters, **{'fwhm': self.fwhm}})


class EllipticalMoffat2D(Fittable2DModel):
    """
    An elliptical 2D Moffat function.

    This function uses a parametrization that is based on the one given
    in the AsPyLib library:
        http://www.aspylib.com/doc/aspylib_fitting.html#elliptical-moffat-psf

    Args:
        mu_x: The x-position of the center of the Moffat function.
        mu_y: The y-position of the center of the Moffat function.
        alpha_x: The alpha_x parameter of the Moffat function.
        alpha_y: The alpha_y parameter of the Moffat function.
        amplitude: The (peak) amplitude of the Moffat function.
        theta: Rotation angle (in radian), measured counter-clockwise.
        offset: Global offset (e.g., the background flux level).
        beta: The beta parameter of the Moffat function. For `beta = 1`,
            the Moffat function is essentially just a Cauchy function
            (also known as Lorentz or Breit-Wigner).
    """

    def __init__(
        self,
        mu_x: float = 0.0,
        mu_y: float = 0.0,
        alpha_x: float = 1.0,
        alpha_y: float = 1.0,
        beta: float = 1.0,
        theta: float = 0.0,
        amplitude: float = 1.0,
        offset: float = 0.0,
    ) -> None:

        self.mu_x = mu_x
        self.mu_y = mu_y
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
        self.beta = beta
        self.theta = theta
        self.amplitude = amplitude
        self.offset = offset

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        return (
            'mu_x',
            'mu_y',
            'alpha_x',
            'alpha_y',
            'beta',
            'theta',
            'amplitude',
            'offset',
        )

    @property
    def fwhm_x(self) -> float:
        """
        Compute the full width half maximum (FWHM) in x-direction.
        """
        return 2 * math.sqrt(2 ** (1 / self.beta) - 1) * self.alpha_x

    @property
    def fwhm_y(self) -> float:
        """
        Compute the full width half maximum (FWHM) in y-direction.
        """
        return 2 * math.sqrt(2 ** (1 / self.beta) - 1) * self.alpha_y

    @staticmethod
    def evaluate(
        meshgrid: Tuple[np.ndarray, np.ndarray],
        *parameters: float,
    ) -> np.ndarray:

        # Unpack the meshgrid as well as the parameters
        xx_grid, yy_grid = meshgrid
        (
            mu_x,
            mu_y,
            alpha_x,
            alpha_y,
            beta,
            theta,
            amplitude,
            offset,
        ) = parameters

        # Compute (x - x_mu) and (y - y_mu)
        x_diff = xx_grid - mu_x
        y_diff = yy_grid - mu_y

        # Compute the coefficients A, B and C
        a = (np.cos(theta) / alpha_x) ** 2 + (np.sin(theta) / alpha_y) ** 2
        b = (np.sin(theta) / alpha_x) ** 2 + (np.cos(theta) / alpha_y) ** 2
        c = (
            2
            * np.sin(theta)
            * np.cos(theta)
            * (1 / alpha_x ** 2 - 1 / alpha_y ** 2)
        )

        # Combine all parts into the final function
        return np.asarray(
            amplitude
            / (
                1
                + (a * x_diff ** 2)
                + (b * y_diff ** 2)
                + (c * x_diff * y_diff)
            )
            ** beta
            + offset
        )

    def __repr__(self) -> str:
        return pformat(
            {
                **self.named_parameters,
                **{'fwhm_x': self.fwhm_x, 'fwhm_y': self.fwhm_y},
            }
        )
