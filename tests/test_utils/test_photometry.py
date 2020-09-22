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
    position = (30, 30)
    frame = np.zeros((101, 101))
    frame = add_array_with_interpolation(frame, gaussian_blob, position)

    # Test case 1
    aperture = CustomCircularAperture(positions=position, r=4)
    results = aperture.get_statistic(data=frame, statistic_function=np.nanmax)
    if isinstance(results, list):
        raise RuntimeError
    assert np.isclose(results, 13)

    # Test case 2
    aperture = CustomCircularAperture(positions=(60, 60), r=4)
    results = aperture.get_statistic(data=frame, statistic_function=np.nanmax)
    if isinstance(results, list):
        raise RuntimeError
    assert np.isclose(results, 0)

    # Test case 2
    aperture = CustomCircularAperture(positions=[position, (60, 60)], r=4)
    results = aperture.get_statistic(data=frame, statistic_function=np.nanmax)
    if isinstance(results, float):
        raise RuntimeError
    assert np.isclose(results[0], 13)
    assert np.isclose(results[1], 0)

    # Test case 4
    aperture = CustomCircularAperture(positions=position, r=8)
    results = aperture.get_statistic(data=frame, statistic_function=np.nansum)
    if isinstance(results, list):
        raise RuntimeError
    assert np.isclose(results, np.sum(gaussian_blob))


def test__fit_2d_gaussian() -> None:

    # Create a 2D Gaussian blob
    gaussian_blob = Gaussian2DKernel(
        x_stddev=2, y_stddev=2, x_size=15, y_size=15,
    ).array
    gaussian_blob /= np.max(gaussian_blob)
    gaussian_blob *= 7

    # Construct test data by adding the blob into an all-zero frame
    position = (30, 30)
    frame = np.zeros((101, 101))
    frame = add_array_with_interpolation(frame, gaussian_blob, position)

    # Test case 1
    aperture = CustomCircularAperture(positions=position, r=4)
    results = aperture.fit_2d_gaussian(data=frame)
    if isinstance(results, list):
        raise RuntimeError
    assert np.isclose(results[0], 7)

    # Test case 2
    aperture = CustomCircularAperture(positions=[position, (60, 60)], r=4)
    results = aperture.fit_2d_gaussian(data=frame)
    if not isinstance(results, list):
        raise RuntimeError
    assert np.isclose(results[0][0], 7)
    assert np.isclose(results[1][0], 0)
