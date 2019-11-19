"""
Provide a few custom extensions to the photometry functions of photutils
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Callable

from scipy.optimize import curve_fit
from photutils import CircularAperture

import numpy as np


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class CustomCircularAperture(CircularAperture):
    """
    A custom extension of photutils's CircularAperture class, which
    provides additional functions to do photometry.
    """

    def get_statistic(self,
                      data: np.ndarray,
                      statistic_function: Callable = np.nansum,
                      method: str = 'exact',
                      subpixels: int = 5):
        """
        Compute a particular summary statistic (such as the pixel sum
        or the maximum) for the aperture.

        Args:
            data: A 2D numpy array containing the data on which the
                aperture will be placed to which the 2D Gaussian is
                fitted.
            statistic_function: A function that is used to compute the
                statistic of interest on the aperture. Must be able to
                process 2D numpy arrays and handle NaN values.
            method: The value for the `method` parameter that is passed
                to the `photutils.CircularAperture.to_mask()` method.
            subpixels: The value for the `subpixels` parameter that is
                passed to the `photutils.CircularAperture.to_mask()`
                method.

        Returns:
            The `statistic_function`, applied to the pixels contained
            in the CustomCircularAperture.
        """

        # Create a mask from the aperture and use it to crop the data
        mask = self.to_mask(method=method, subpixels=subpixels)
        cropped_data = mask.cutout(data, copy=True)

        # Set everything outside of the masked area to NaN. There seems to
        # exist no better way to do this if we want to avoid "taking into
        # account edge effects" (i.e, multiplying the edges of the aperture
        # with some factor F with 0 < F < 1), which would bias the calculation
        # of the statistics_function to too low values.
        # noinspection PyProtectedMember
        # pylint: disable=protected-access
        cropped_data[mask._mask] = np.nan

        # Return the statistic_function for the cropped, masked data
        return statistic_function(cropped_data)

    def fit_2d_gaussian(self,
                        data: np.ndarray,
                        method: str = 'exact',
                        subpixels: int = 5):
        """
        Fit a simple, symmetric 2D Gauss with only two parameters (i.e.,
        the amplitude and standard deviation) to the center of the
        aperture and return the fit parameters.

        This is possibly a more robust way to estimate the "maximum" of
        the aperture, rather than simply taking a pixel-wise maximum.

        Args:
            data: A 2D numpy array containing the data on which the
                aperture will be placed to which the 2D Gaussian is
                fitted.
            method: The value for the `method` parameter that is passed
                to the `photutil.CircularAperture.to_mask()` method.
            subpixels: The value for the `subpixels` parameter that is
                passed to the `photutil.CircularAperture.to_mask()`
                method.

        Returns:
            A numpy array containing the parameters for the 2D Gaussian
            returned by the fit. Should be `(amplitude, sigma)`.
        """

        # Create a mask from the aperture and use it to crop the data
        mask = self.to_mask(method=method, subpixels=subpixels)
        cropped_data = mask.cutout(data, copy=True)

        # Compute center of the cropped data
        center = list(_ / 2 - 0.5 for _ in cropped_data.shape)

        # Get the indices of positions inside the aperture
        idx = mask.data.astype(bool).reshape(-1,)

        # Use the indices to select the data which we can use for the fit
        xdata = np.indices(cropped_data.shape).reshape(2, -1)[:, idx]
        ydata = cropped_data.reshape(-1,)[idx]

        # Define a 2D Gauss function using the value for center computed above
        def gauss2d(x, amplitude, sigma):
            inner = ((x[0] - center[0]) ** 2 / (2 * sigma) ** 2 +
                     (x[1] - center[1]) ** 2 / (2 * sigma) ** 2)
            return amplitude * np.exp(-inner)

        # Define the bounds and the initial values for the fit
        abs_max = np.nanmax(np.abs(cropped_data))
        bounds = [(-abs_max, 1), (abs_max, np.inf)]
        p0 = [abs_max, 1]

        # Fit the 2D Gaussian to the cropped data, and return the optimal
        # parameters according to the fit (ignore the covariance matrix)
        parameters, _ = curve_fit(f=gauss2d,
                                  xdata=xdata,
                                  ydata=ydata,
                                  bounds=bounds,
                                  p0=p0,
                                  xtol=0.001,
                                  ftol=0.001)
        return parameters
