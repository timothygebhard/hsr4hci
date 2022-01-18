"""
Tests for metrics.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pprint import pprint

from astropy.modeling import models
from astropy.units import Quantity
from skimage.filters import gaussian

import numpy as np
import pytest

from hsr4hci.metrics import two_sample_t_test, compute_metrics


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__two_sample_t_test() -> None:

    # Case 1
    signal, noise, snr, fpf, p_value = two_sample_t_test(
        planet_samples=[5],
        noise_samples=[-1, 1, 0, -1, 1],
    )
    assert signal == 5
    assert noise == np.sqrt(1 + 1 / 5)
    assert np.isclose(snr, 5 / np.sqrt(1 + 1 / 5))
    assert np.isclose(fpf, 0.0051523818078138195)
    assert np.isclose(p_value, 0.9948476181921861)

    # Case 2
    with pytest.raises(ValueError) as value_error:
        two_sample_t_test(planet_samples=[], noise_samples=[1, 2])
    assert 'planet_samples must have at least 1' in str(value_error)

    # Case 3
    with pytest.raises(ValueError) as value_error:
        two_sample_t_test(planet_samples=[1], noise_samples=[1])
    assert 'noise_samples must have at least 2' in str(value_error)


def test__compute_metrics() -> None:

    np.random.seed(42)

    # Create a fake signal estimate
    frame_size = (65, 65)
    signal_estimate = np.random.normal(0, 0.5, frame_size)
    signal_estimate = gaussian(signal_estimate, sigma=5)
    x, y = np.meshgrid(np.arange(frame_size[0]), np.arange(frame_size[1]))
    model = models.Gaussian2D(x_mean=48, x_stddev=2, y_mean=32, y_stddev=2)
    signal_estimate += np.asarray(model(x, y))

    # Case 1
    results, positions = compute_metrics(
        frame=signal_estimate,
        polar_position=(Quantity(16, 'pixel'), Quantity(270, 'degree')),
        planet_mode='FS',
        noise_mode='P',
        aperture_radius=Quantity(2.35, 'pixel'),
        search_radius=Quantity(1, 'pixel'),
        exclusion_angle=None,
        n_rotation_steps=10,
    )
    assert np.isclose(
        positions['final']['polar'][0].to('pixel').value, 16.01086125
    )
    assert np.isclose(
        positions['final']['polar'][1].to('radian').value, -1.57203488
    )
    assert np.isclose(positions['final']['cartesian'][0], 48.01084897383275)
    assert np.isclose(positions['final']['cartesian'][1], 31.98016968443035)

    print()
    pprint(results)

    assert np.isclose(results['fpf']['max'], 5.164200653596613e-17)
    assert np.isclose(results['fpf']['mean'], 3.511819194376848e-17)
    assert np.isclose(results['fpf']['median'], 3.834279134370597e-17)
    assert np.isclose(results['fpf']['min'], 5.535349294883412e-18)
    assert np.isclose(results['fpf']['std'], 1.155278577366806e-17)
    assert np.isclose(results['log_fpf']['max'], 17.256854968813425)
    assert np.isclose(results['log_fpf']['mean'], 16.502279822495144)
    assert np.isclose(results['log_fpf']['median'], 16.416316273799083)
    assert np.isclose(results['log_fpf']['min'], 16.28699689168509)
    assert np.isclose(results['log_fpf']['std'], 0.2506379199885584)
    assert np.isclose(results['noise']['max'], 0.021591052642817854)
    assert np.isclose(results['noise']['mean'], 0.021125651139223087)
    assert np.isclose(results['noise']['median'], 0.021179069594017063)
    assert np.isclose(results['noise']['min'], 0.020778807014472208)
    assert np.isclose(results['noise']['std'], 0.00025066276461067156)
    assert np.isclose(results['p_value']['max'], 1.0)
    assert np.isclose(results['p_value']['mean'], 1.0)
    assert np.isclose(results['p_value']['median'], 1.0)
    assert np.isclose(results['p_value']['min'], 0.9999999999999999)
    assert np.isclose(results['p_value']['std'], 5.797950651443767e-17)
    assert np.isclose(results['signal']['max'], 0.7742381617783726)
    assert np.isclose(results['signal']['mean'], 0.7735469609874865)
    assert np.isclose(results['signal']['median'], 0.7735416641351142)
    assert np.isclose(results['signal']['min'], 0.7729656408388326)
    assert np.isclose(results['signal']['std'], 0.0003730376838223088)
    assert np.isclose(results['snr']['max'], 37.19971220197919)
    assert np.isclose(results['snr']['mean'], 36.62148319066654)
    assert np.isclose(results['snr']['median'], 36.52914588278508)
    assert np.isclose(results['snr']['min'], 35.84790987208666)
    assert np.isclose(results['snr']['std'], 0.4219717303747466)

    # Case 2
    with pytest.raises(RuntimeError) as runtime_error:
        compute_metrics(
            frame=signal_estimate,
            polar_position=(Quantity(2.5, 'pixel'), Quantity(270, 'degree')),
            planet_mode='FS',
            noise_mode='P',
            aperture_radius=Quantity(2.35, 'pixel'),
            search_radius=Quantity(1, 'pixel'),
            exclusion_angle=None,
            n_rotation_steps=10,
        )
    assert 'Too few reference positions' in str(runtime_error)
