"""
Utility functions for fitting (e.g., PSFs with analytical models).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Callable, Sequence, Tuple

from scipy.optimize import curve_fit

import numpy as np


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def moffat_2d(
    meshgrid: Tuple[np.ndarray, np.ndarray],
    x_center: float = 0.0,
    y_center: float = 0.0,
    fwhm_x: float = 1.0,
    fwhm_y: float = 1.0,
    amplitude: float = 1.0,
    offset: float = 0.0,
    theta: float = 0.0,
    beta: float = 1.0,
) -> np.ndarray:
    """
    A 2D elliptical Moffat function that can be used to fit a PSF.

    This function uses a parametrization based on the one given in the
    AsPyLib library:
        http://www.aspylib.com/doc/aspylib_fitting.html#elliptical-moffat-psf

    Args:
        meshgrid: A numpy meshgrid, i.e., a tuple of two 2D arrays that
            contain the positions where to compute the Moffat function.
        x_center: The x-position of the center of the Moffat function.
        y_center: The x-position of the center of the Moffat function.
        fwhm_x: The FWHM along the x-axis (in pixels).
        fwhm_y: The FWHM along the y-axis (in pixels).
        amplitude: The peak amplitude of the Moffat function.
        theta: Rotation angle (in radian), measured counter-clockwise.
        offset: Global offset, e.g., the background flux level.
        beta: The beta parameter of the Moffat function. For beta = 1,
            the Moffat function is essentially just a Cauchy (also known
            as Lorentz or Breit-Wigner) function.

    Returns:
        A 2D numpy array with the values of the Moffat function computed
        at the position specified in the input `meshgrid`.
    """

    # Unpack the meshgrid into the grid for the x and y positions
    (xx_grid, yy_grid) = meshgrid

    # Compute (x - x_0) and (y - y_0)
    x_diff = xx_grid - x_center
    y_diff = yy_grid - y_center

    # Convert the FWHM into the alpha parameter of the Moffat function
    alpha_x = 0.5 * fwhm_x / np.sqrt(2**(1 / beta) - 1)
    alpha_y = 0.5 * fwhm_y / np.sqrt(2**(1 / beta) - 1)

    # Compute the coefficients A, B and C
    a = (np.cos(theta) / alpha_x)**2 + (np.sin(theta) / alpha_y)**2
    b = (np.sin(theta) / alpha_x)**2 + (np.cos(theta) / alpha_y)**2
    c = 2 * np.sin(theta) * np.cos(theta) * (1 / alpha_x**2 - 1 / alpha_y**2)

    # Combine all parts into the final function
    moffat = offset + amplitude / (1 +
                                   a * x_diff**2 +
                                   b * y_diff**2 +
                                   c * x_diff * y_diff) ** beta

    return moffat


def fit_2d_function(
    frame: np.ndarray,
    function: Callable,
    p0: Sequence,
) -> np.ndarray:
    """
    Fit a given 2D function to a frame (e.g., fit a PSF with a Moffat).

    Args:
        frame: A 2D numpy array containing which is used as the target
            for the fit.
        function: The function to be fitted to frame. This function must
            be a 2D scalar function `f: R^2 -> R` which can deal with
            vector inputs, that it, should take a meshgrid with the X
            and Y coordinates that match the given `frame` as input,
            as well as some parameters to be optimized. It should then
            also return a 2D numpy array with the same shape as the
            coordinate meshgrid. See moffat_2d() above for an example.
        p0: An array or a list containing the initial guesses for the
            free parameters of the `function` (i.e., those parameters
            that are optimized by fitting the function to the frame).

    Returns:
        An array with the same shape as `p0` which contains the
        optimized parameters of the function obtained from the fit.
    """

    # Construct a meshgrid for the positions in the frame.
    # This should be consistent with the astropy / photutils conventions for
    # pixel coordinates, where the lower left hand corner of an image is at
    # (-0.5, -0.5), meaning the the center of the lower left hand corner pixel
    # is at (0, 0).
    x_grid = np.arange(0, frame.shape[0])
    y_grid = np.arange(0, frame.shape[1])
    meshgrid = np.meshgrid(x_grid, y_grid)

    # Define dummy function to ravel the output of the target function
    def target_function(*args: Any, **kwargs: Any) -> np.ndarray:
        return function(*args, **kwargs).ravel()

    # Fit the frame using the target function and the initial parameter guess
    with np.warnings.catch_warnings():

        # Ignore some numpy warnings here that can happen if the minimizer
        # explores a particularly bad parameter range.
        np.warnings.filterwarnings('ignore', r'invalid value encountered in')
        np.warnings.filterwarnings('ignore', r'overflow encountered in')

        # Actually perform the fit using the Levenberg-Marquardt algorithm
        p_opt, _ = curve_fit(f=target_function,
                             xdata=meshgrid,
                             ydata=np.nan_to_num(frame).ravel(),
                             p0=p0)

    return p_opt
