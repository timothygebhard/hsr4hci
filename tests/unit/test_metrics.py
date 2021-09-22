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
        positions['final']['polar'][0].to('pixel').value, 16.01080024421563
    )
    assert np.isclose(
        positions['final']['polar'][1].to('radian').value, -1.5720312352633194
    )
    assert np.isclose(results['fpf']['max'], 4.985079950867889e-17)
    assert np.isclose(results['fpf']['mean'], 3.389508968147001e-17)
    assert np.isclose(results['fpf']['median'], 3.7016290701829294e-17)
    assert np.isclose(results['fpf']['min'], 5.3292228535917906e-18)
    assert np.isclose(results['fpf']['std'], 1.1155016447011583e-17)
    assert np.isclose(results['log_fpf']['max'], 17.282327941119874)
    assert np.isclose(results['log_fpf']['mean'], 16.526431865957733)
    assert np.isclose(results['log_fpf']['median'], 16.44033950338373)
    assert np.isclose(results['log_fpf']['min'], 16.310988015553853)
    assert np.isclose(results['log_fpf']['std'], 0.25104035793468965)
    assert np.isclose(results['noise']['max'], 0.02159086332484625)
    assert np.isclose(results['noise']['mean'], 0.021125462204945204)
    assert np.isclose(results['noise']['median'], 0.02117884744469653)
    assert np.isclose(results['noise']['min'], 0.02077860767195894)
    assert np.isclose(results['noise']['std'], 0.0002506598504073832)
    assert np.isclose(results['p_value']['max'], 1.0)
    assert np.isclose(results['p_value']['mean'], 1.0)
    assert np.isclose(results['p_value']['median'], 1.0)
    assert np.isclose(results['p_value']['min'], 0.9999999999999999)
    assert np.isclose(results['p_value']['std'], 4.734006883291518e-17)
    assert np.isclose(results['signal']['max'], 0.7769396634948865)
    assert np.isclose(results['signal']['mean'], 0.7762484349890049)
    assert np.isclose(results['signal']['median'], 0.7762431434540997)
    assert np.isclose(results['signal']['min'], 0.775667102956501)
    assert np.isclose(results['signal']['std'], 0.0003730524628884189)
    assert np.isclose(results['snr']['max'], 37.330080783193)
    assert np.isclose(results['snr']['mean'], 36.7497062463769)
    assert np.isclose(results['snr']['median'], 36.65708303670411)
    assert np.isclose(results['snr']['min'], 35.97334674314541)
    assert np.isclose(results['snr']['std'], 0.42348499291612113)

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
