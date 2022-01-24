"""
Tests for metrics.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from astropy.modeling import models
from astropy.units import Quantity
from deepdiff import DeepDiff
from skimage.filters import gaussian

import numpy as np
import pytest

from hsr4hci.metrics import two_sample_t_test, compute_metrics


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------


def test__two_sample_t_test() -> None:
    """
    Test `hsr4hci.metrics.two_sample_t_test`.
    """

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
    """
    Test `hsr4hci.metrics.compute_metrics`.
    """

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

    targets = {
        'fpf': {
            'max': 6.194854293428218e-17,
            'mean': 4.214132959935859e-17,
            'median': 4.60012631087341e-17,
            'min': 6.715709917068221e-18,
            'std': 1.3840535988546529e-17,
        },
        'log_fpf': {
            'max': 17.172908071339435,
            'mean': 16.422724601964216,
            'median': 16.337230243239688,
            'min': 16.207968904026302,
            'std': 0.24930074956372117,
        },
        'noise': {
            'max': 0.021590863324846252,
            'mean': 0.021125462204945204,
            'median': 0.02117884744469653,
            'min': 0.02077860767195894,
            'std': 0.0002506598504073832,
        },
        'p_value': {
            'max': 1.0,
            'mean': 1.0,
            'median': 1.0,
            'min': 0.9999999999999999,
            'std': 7.485122105058051e-17,
        },
        'signal': {
            'max': 0.76537519658579,
            'mean': 0.7646839680799083,
            'median': 0.7646786765450032,
            'min': 0.7641026360474045,
            'std': 0.0003730524628884189,
        },
        'snr': {
            'max': 36.7735243915584,
            'mean': 36.20221102207236,
            'median': 36.11104449804924,
            'min': 35.43772819306231,
            'std': 0.41701411203610933,
        },
    }
    assert not DeepDiff(results, targets)

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
