"""
Tests for contrast.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from astropy.modeling import models
from astropy.units import Quantity
from scipy.stats import norm

import numpy as np
import pandas as pd

from hsr4hci.general import shift_image, crop_or_pad
from hsr4hci.contrast import get_contrast, get_contrast_curve


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
    assert np.isclose(
        results['observed_flux_ratio'], 0.010587254186049756, rtol=1e-4
    )
    assert np.isclose(
        results['observed_contrast'], 4.938041649889568, rtol=1e-4
    )
    assert np.isclose(results['throughput'], 1.0587254186049757, rtol=1e-4)

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


def test_get_contrast_curve() -> None:

    # Case 1
    df = pd.DataFrame(
        {
            'separation': 5 * np.ones(11),
            'expected_contrast': np.linspace(5, 15, 11),
            'fpf_mean': (
                2
                * (1 - norm.cdf(5, 0, 1))
                * np.tanh(7.5 - 0.5 * np.arange(5, 16))
            ),
        }
    )
    separations, detection_limits = get_contrast_curve(df, 5, False)
    assert np.array_equal(separations, np.array([5]))
    assert np.allclose(
        detection_limits, np.array([15 - 2 * np.arctanh(0.5)]), atol=0.05
    )

    # Case 2
    df = pd.DataFrame(
        {
            'separation': 5 * np.ones(11),
            'expected_contrast': np.linspace(5, 15, 11),
            'fpf_mean': (
                (1 - norm.cdf(5, 0, 1))
                ** (2 * np.tanh(7.5 - 0.5 * np.arange(5, 16)))
            ),
        }
    )
    separations, detection_limits = get_contrast_curve(df, 5, True)
    assert np.array_equal(separations, np.array([5]))
    assert np.allclose(
        detection_limits, np.array([15 - 2 * np.arctanh(0.5)]), atol=0.05
    )
