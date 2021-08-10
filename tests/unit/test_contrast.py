"""
Tests for contrast.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from astropy.modeling import models
from astropy.units import Quantity

import numpy as np

from hsr4hci.general import shift_image, crop_or_pad
from hsr4hci.contrast import get_contrast


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test_get_contrast() -> None:

    np.random.seed(42)

    # Define shortcuts
    frame_size = (101, 101)

    # Create a fake PSF template
    x, y = np.meshgrid(np.arange(33), np.arange(33))
    gaussian = models.Gaussian2D(x_mean=16, y_mean=16)
    psf_template = np.asarray(gaussian(x, y))
    psf_template /= np.max(psf_template)
    psf_template *= 100
    psf_resized = crop_or_pad(psf_template, frame_size)

    # Case 1
    signal_estimate = shift_image(psf_resized, (23, 0))
    results = get_contrast(
        signal_estimate=signal_estimate,
        polar_position=(Quantity(23, 'pixel'), Quantity(270, 'degree')),
        psf_template=psf_template,
        metadata={'DIT_STACK': 1, 'DIT_PSF_TEMPLATE': 1, 'ND_FILTER': 1},
        no_fake_planets=None,
        expected_contrast=None,
    )
    assert np.isclose(results['observed_flux_ratio'], 1)
    assert np.isclose(results['observed_contrast'], 0)
    assert np.isnan(results['throughput'])

    # Case 2
    signal_estimate = shift_image(psf_resized / 100, (0, 37))
    results = get_contrast(
        signal_estimate=signal_estimate,
        polar_position=(Quantity(37, 'pixel'), Quantity(0, 'degree')),
        psf_template=psf_template,
        metadata={'DIT_STACK': 1, 'DIT_PSF_TEMPLATE': 1, 'ND_FILTER': 1},
        no_fake_planets=None,
        expected_contrast=5,
    )
    assert np.isclose(results['observed_flux_ratio'], 0.01)
    assert np.isclose(
        results['observed_contrast'], results['expected_contrast']
    )
    assert np.isclose(results['throughput'], 1)

    # Case 3
    signal_estimate = shift_image(psf_resized / 100, (-10, 0))
    no_fake_planets = np.random.normal(0, 1, signal_estimate.shape)
    signal_estimate += no_fake_planets
    results = get_contrast(
        signal_estimate=signal_estimate,
        polar_position=(Quantity(10, 'pixel'), Quantity(90, 'degree')),
        psf_template=psf_template,
        metadata={'DIT_STACK': 1, 'DIT_PSF_TEMPLATE': 1, 'ND_FILTER': 1},
        no_fake_planets=no_fake_planets,
        expected_contrast=5,
    )
    assert np.isclose(results['observed_flux_ratio'], 0.01)
    assert np.isclose(
        results['observed_contrast'], results['expected_contrast']
    )
    assert np.isclose(results['throughput'], 1)

    # Case 4
    signal_estimate = shift_image(psf_resized / 100, (-33, 0))
    no_fake_planets = np.random.normal(0, 0.1, signal_estimate.shape)
    signal_estimate += no_fake_planets
    results = get_contrast(
        signal_estimate=signal_estimate,
        polar_position=(Quantity(33, 'pixel'), Quantity(90, 'degree')),
        psf_template=psf_template,
        metadata={'DIT_STACK': 1, 'DIT_PSF_TEMPLATE': 1, 'ND_FILTER': 1},
        no_fake_planets=None,
        expected_contrast=5,
    )
    assert np.isclose(results['observed_flux_ratio'], 0.01057930735268089)
    assert np.isclose(results['observed_contrast'], 4.93885691363388)
    assert np.isclose(results['throughput'], 1.0579307352680893)

    # Case 5
    signal_estimate = np.zeros(frame_size)
    results = get_contrast(
        signal_estimate=signal_estimate,
        polar_position=(Quantity(33, 'pixel'), Quantity(90, 'degree')),
        psf_template=psf_template,
        metadata={'DIT_STACK': 1, 'DIT_PSF_TEMPLATE': 1, 'ND_FILTER': 1},
        no_fake_planets=None,
        expected_contrast=0,
    )
    assert np.isclose(results['observed_flux_ratio'], 0)
    assert np.isinf(results['observed_contrast'])
    assert np.isclose(results['throughput'], 0)
