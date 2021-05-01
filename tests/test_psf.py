"""
Tests for psf.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from astropy.units import Quantity

import numpy as np

from hsr4hci.general import crop_center
from hsr4hci.psf import get_psf_fwhm, get_artificial_psf


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__get_artificial_psf() -> None:

    artificial_psf = get_artificial_psf(
        pixscale=Quantity(0.0271, 'arcsec / pixel'),
        lambda_over_d=Quantity(0.0956, 'arcsec'),
    )

    assert artificial_psf.ndim == 2
    assert artificial_psf.shape == (41, 41)
    assert np.isclose(np.sum(artificial_psf), 1)


def test__get_psf_fwhm() -> None:

    psf_template = get_artificial_psf(
        pixscale=Quantity(0.0271, 'arcsec / pixel'),
        lambda_over_d=Quantity(0.0956, 'arcsec'),
    )

    # Test case 1
    psf_fwhm = get_psf_fwhm(psf_template=psf_template)
    assert np.isclose(psf_fwhm, 3.86723)

    # Test case 2
    psf_cropped = crop_center(psf_template, (31, 31))
    psf_fwhm = get_psf_fwhm(psf_template=psf_cropped)
    assert np.isclose(psf_fwhm, 3.86600)
