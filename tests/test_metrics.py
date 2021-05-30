"""
Tests for metrics.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

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

    signal, noise, snr, fpf, p_value = two_sample_t_test(
        planet_samples=[5],
        noise_samples=[-1, 1, 0, -1, 1],
    )
    assert signal == 5
    assert noise == np.sqrt(1 + 1 / 5)
    assert np.isclose(snr, 5 / np.sqrt(1 + 1 / 5))
    assert np.isclose(fpf, 0.0051523818078138195)
    assert np.isclose(p_value, 0.9948476181921861)


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
        positions['final']['polar'][0].to('pixel').value, 16.01104586
    )
    assert np.isclose(
        positions['final']['polar'][1].to('radian').value, -1.57211132
    )
    assert np.isclose(results['fpf']['max'], 4.985079950867889e-17)
    assert np.isclose(results['fpf']['mean'], 3.389508968147001e-17)
    assert np.isclose(results['fpf']['median'], 3.7016290701829294e-17)
    assert np.isclose(results['fpf']['min'], 5.3292228535917906e-18)
    assert np.isclose(results['fpf']['std'], 1.1155016447011583e-17)
    assert np.isclose(results['log_fpf']['max'], 17.27333611836642)
    assert np.isclose(results['log_fpf']['mean'], 16.517762842231445)
    assert np.isclose(results['log_fpf']['median'], 16.43160710282887)
    assert np.isclose(results['log_fpf']['min'], 16.302327872068012)
    assert np.isclose(results['log_fpf']['std'], 0.2509437715244665)
    assert np.isclose(results['noise']['max'], 0.021591978585007096)
    assert np.isclose(results['noise']['mean'], 0.02112641155509073)
    assert np.isclose(results['noise']['median'], 0.021180148095241408)
    assert np.isclose(results['noise']['min'], 0.02077980551946452)
    assert np.isclose(results['noise']['std'], 0.00025072540799743036)
    assert np.isclose(results['p_value']['max'], 1.0)
    assert np.isclose(results['p_value']['mean'], 1.0)
    assert np.isclose(results['p_value']['median'], 1.0)
    assert np.isclose(results['p_value']['min'], 0.9999999999999999)
    assert np.isclose(results['p_value']['std'], 4.734006883291518e-17)
    assert np.isclose(results['signal']['max'], 0.7760009431476597)
    assert np.isclose(results['signal']['mean'], 0.7753099579273356)
    assert np.isclose(results['signal']['median'], 0.7753044468190109)
    assert np.isclose(results['signal']['min'], 0.7747286941495293)
    assert np.isclose(results['signal']['std'], 0.00037295309654538256)
    assert np.isclose(results['snr']['max'], 37.282769245546504)
    assert np.isclose(results['snr']['mean'], 36.70362888897134)
    assert np.isclose(results['snr']['median'], 36.61052751305583)
    assert np.isclose(results['snr']['min'], 35.92801806912074)
    assert np.isclose(results['snr']['std'], 0.4230403938588033)


    # Case 2
    with pytest.raises(RuntimeError) as runtime_error:
        compute_metrics(
            frame=signal_estimate,
            polar_position=(Quantity(2, 'pixel'), Quantity(270, 'degree')),
            planet_mode='FS',
            noise_mode='P',
            aperture_radius=Quantity(2.35, 'pixel'),
            search_radius=Quantity(1, 'pixel'),
            exclusion_angle=None,
            n_rotation_steps=10,
        )
    assert 'Too few reference positions' in str(runtime_error)
