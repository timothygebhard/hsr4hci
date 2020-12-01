"""
Tests for photometry.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from astropy.convolution import Gaussian2DKernel

import numpy as np

from hsr4hci.utils.general import add_array_with_interpolation
from hsr4hci.utils.photometry import CustomCircularAperture


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__get_statistic() -> None:

    # Create a 2D Gaussian blob
    gaussian_blob = Gaussian2DKernel(
        x_stddev=2, y_stddev=2, x_size=15, y_size=15,
    ).array
    gaussian_blob /= np.max(gaussian_blob)
    gaussian_blob *= 13

    # Construct test data by adding the blob into an all-zero frame
    # Note that the position must be something like (x.5, y.5) so we do not
    # actually use bilinear interpolation, which might change the peak value
    # in a single pixel. Also, we have to flip the order of the position when
    # we use it to actually access a numpy array!
    position = (30.5, 20.5)
    frame = np.zeros((101, 101))
    frame = add_array_with_interpolation(frame, gaussian_blob, position[::-1])

    # Test case 1: check if peak of Gaussian is recovered correctly
    aperture = CustomCircularAperture(positions=position, r=4)
    results = aperture.get_statistic(data=frame, statistic_function=np.nanmax)
    assert isinstance(results, float)
    assert np.isclose(results, 13)

    # Test case 2: check if we get a 0 when we should get a 0
    aperture = CustomCircularAperture(positions=(60, 60), r=4)
    results = aperture.get_statistic(data=frame, statistic_function=np.nanmax)
    assert isinstance(results, float)
    assert np.isclose(results, 0)

    # Test case 3: check if we get the correct results for multiple positions
    aperture = CustomCircularAperture(positions=[position, (60, 60)], r=4)
    results = aperture.get_statistic(data=frame, statistic_function=np.nanmax)
    assert isinstance(results, list)
    assert np.isclose(results[0], 13)
    assert np.isclose(results[1], 0)

    # Test case 4: check if total flux in an aperture is recovered correctly
    aperture = CustomCircularAperture(positions=position, r=10)
    results = aperture.get_statistic(data=frame, statistic_function=np.nansum)
    assert isinstance(results, float)
    assert np.isclose(results, np.sum(gaussian_blob))


def test__fit_2d_gaussian() -> None:

    # Create a 2D Gaussian blob
    gaussian_blob = Gaussian2DKernel(
        x_stddev=2, y_stddev=2, x_size=15, y_size=15,
    ).array

    # Construct test data by adding the blob into an all-zero frame
    position = (30, 20)
    frame = np.zeros((101, 101))
    frame = add_array_with_interpolation(frame, gaussian_blob, position[::-1])
    frame /= np.max(frame)
    frame *= 7

    # Test case 1
    aperture = CustomCircularAperture(positions=position, r=4)
    results = aperture.fit_2d_gaussian(data=frame)
    assert isinstance(results, tuple)
    assert np.isclose(results[0], 7, atol=0.01)

    # Test case 2
    aperture = CustomCircularAperture(positions=[position, (60, 60)], r=4)
    results = aperture.fit_2d_gaussian(data=frame)
    assert isinstance(results, list)
    assert np.isclose(results[0][0], 7, atol=0.01)
    assert np.isclose(results[1][0], 0)
