"""
Tests for psf.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from astropy.modeling import models

import numpy as np

from hsr4hci.psf import get_psf_fwhm


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__get_psf_fwhm() -> None:
    """
    Test `hsr4hci.psf.get_psf_fwhm`.
    """

    # Case 1
    x, y = np.meshgrid(np.arange(33), np.arange(33))
    gaussian = models.Gaussian2D(x_mean=16, y_mean=16)
    psf_template = np.asarray(gaussian(x, y))
    actual_psf_fwhm = get_psf_fwhm(psf_template=psf_template)
    expected_psf_fwhm = 2 * np.sqrt(2 * np.log(2))
    assert np.isclose(actual_psf_fwhm, expected_psf_fwhm)

    # Test case 2
    x, y = np.meshgrid(np.arange(53), np.arange(53))
    gaussian = models.Gaussian2D(x_mean=26, y_mean=26, x_stddev=2, y_stddev=2)
    psf_template = np.asarray(gaussian(x, y))
    actual_psf_fwhm = get_psf_fwhm(psf_template=psf_template)
    expected_psf_fwhm = 2 * 2 * np.sqrt(2 * np.log(2))
    assert np.isclose(actual_psf_fwhm, expected_psf_fwhm)
