"""
Provide a few custom extensions to the photometry functions of photutils
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Callable, List, Tuple, Union

from scipy.optimize import curve_fit
from photutils import CircularAperture

import bottleneck as bn
import numpy as np


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class CustomCircularAperture(CircularAperture):
    """
    A custom extension of photutils's CircularAperture class, which
    provides additional functions to do photometry.
    """

    def get_statistic(
        self,
        data: np.ndarray,
        statistic_function: Callable = bn.nansum,
        method: str = 'exact',
        subpixels: int = 5,
    ) -> Union[float, List[float]]:
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

        # Initialize list results
        results = []

        # Create a mask from the aperture and use it to crop the data
        masks = self.to_mask(method=method, subpixels=subpixels)

        # Make sure that masks is always a list. This is necessary because
        # self.to_mask() may return either a single mask, or a list of masks,
        # depending on whether the `CustomCircularAperture` in instantiated
        # using a single or multiple `positions`.
        if not isinstance(masks, list):
            masks = [masks]

        # Loop over all masks to collect results
        for mask in masks:

            # Crop the data using the (current) circular aperture mask
            cropped_data = mask.cutout(data, copy=True)

            # Set everything outside of the masked area to NaN. There seems to
            # exist no better way to do this if we want to avoid "taking into
            # account edge effects" (i.e, multiplying the edges of the aperture
            # with some factor F with 0 < F < 1), which would bias the
            # calculation of the statistics_function to too low values.
            # noinspection PyProtectedMember
            # pylint: disable=protected-access
            cropped_data[mask._mask] = np.nan

            # Compute and store the statistic_function for the cropped and
            # masked data
            results.append(float(statistic_function(cropped_data)))

        # Either return a list of values, or a single value
        if len(results) > 1:
            return results
        return results[0]

    def fit_2d_gaussian(
        self, data: np.ndarray, method: str = 'exact', subpixels: int = 5
    ) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
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
            A tuple, or a list of tuples (in case multiples `positions`
            were given), containing the parameters for the 2D Gaussian
            returned by the fit: `(amplitude, sigma)`.
        """

        # Initialize list results
        results = []

        # Create a mask from the aperture and use it to crop the data
        masks = self.to_mask(method=method, subpixels=subpixels)

        # Make sure that masks is always a list (see `get_statistic()`)
        if not isinstance(masks, list):
            masks = [masks]

        # Loop over all masks and fit their contents with a 2D Gaussian
        for mask in masks:

            cropped_data = mask.cutout(data, copy=True)

            # Compute center of the cropped data
            # Note: The -0.5 is necessary, because of the following reason:
            # The size of the bounding box of the aperture is always odd, say
            # 11 by 11 pixels. The center of this now would be (5.5, 5.5).
            # However, the grid of positions that we create by np.indices()
            # below starts at 0, which means the x and y positions are
            # [0, 1, ..., 10]. The correct center for this grid is actually
            # (5, 5) -- hence the correction by -0.5.
            center = (
                cropped_data.shape[0] / 2 - 0.5,
                cropped_data.shape[1] / 2 - 0.5,
            )

            # Get the indices of positions inside the aperture
            idx = mask.data.astype(bool).ravel()

            # Use the indices to select the data which we can use for the fit
            xdata = np.indices(cropped_data.shape).reshape(2, -1)[:, idx]
            ydata = cropped_data.ravel()[idx]

            # Define a 2D Gauss function using the center value computed above
            def gauss2d(
                x: Tuple[float, float], amplitude: float, sigma: float
            ) -> np.ndarray:
                inner = (x[0] - center[0]) ** 2 / (2 * sigma) ** 2 + (
                    x[1] - center[1]
                ) ** 2 / (2 * sigma) ** 2
                return np.asarray(amplitude * np.exp(-inner))

            # Define the bounds and the initial values for the fit
            # Note: The `+ 1` is needed to avoid errors when trying to fit a
            # region where `abs_max` otherwise would be 0.
            abs_max = bn.nanmax(np.abs(cropped_data)) + 1
            bounds = [(-abs_max, 1), (abs_max, np.inf)]
            p0 = [abs_max, 1]

            # Fit the 2D Gaussian to the cropped data, and return the optimal
            # parameters according to the fit (ignore the covariance matrix)
            parameters, _ = curve_fit(
                f=gauss2d,
                xdata=xdata,
                ydata=ydata,
                bounds=bounds,
                p0=p0,
                xtol=0.001,
                ftol=0.001,
            )

            # Unpack the parameters and return them as a tuple
            amplitude, sigma = parameters
            results.append((amplitude, sigma))

        # Either return a list of values, or a single value
        if len(results) > 1:
            return results
        return results[0]
