"""
Tests for match_fraction.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from astropy.modeling import models
from astropy.units import Quantity

import numpy as np

from hsr4hci.forward_modeling import add_fake_planet
from hsr4hci.match_fraction import (
    get_all_match_fractions,
    get_match_fraction_for_position,
)


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__get_match_fraction_for_position() -> None:

    # Create fake PSF template
    x, y = np.meshgrid(np.arange(33), np.arange(33))
    model = models.Gaussian2D(x_mean=17, y_mean=17, x_stddev=2, y_stddev=2)
    psf_template = model(x, y)
    psf_template[psf_template < 0.1] = 0

    n_frames, x_size, y_size = (100, 33, 33)
    frame_size = (x_size, y_size)
    parang = np.linspace(-45, 45, n_frames)

    # Create fake minimal results
    residuals = add_fake_planet(
        stack=np.zeros((n_frames, x_size, y_size)),
        parang=parang,
        psf_template=psf_template,
        polar_position=(Quantity(10, 'pixel'), Quantity(0, 'degree')),
        magnitude=1,
        extra_scaling=1,
        dit_stack=1,
        dit_psf_template=1,
        return_planet_positions=False,
    )
    results = {'50': {'residuals': residuals}}

    signal_times = np.array(
        sorted(list(map(int, filter(lambda _: _.isdigit(), results.keys()))))
    )

    # Case 1
    mean_mf, median_mf, affected_pixels = get_match_fraction_for_position(
        position=(26, 16),
        hypothesis=np.nan,
        results=results,
        parang=parang,
        psf_template=psf_template,
        signal_times=signal_times,
        frame_size=frame_size,
    )
    assert mean_mf == 0
    assert median_mf == 0
    assert np.array_equal(affected_pixels, np.full(frame_size, False))

    # Case 2
    mean_mf, median_mf, affected_pixels = get_match_fraction_for_position(
        position=(26, 16),
        hypothesis=int(n_frames / 2),
        results=results,
        parang=parang,
        psf_template=psf_template,
        signal_times=signal_times,
        frame_size=frame_size,
    )
    assert np.isclose(mean_mf, 0.9996813941131175)
    assert np.isclose(median_mf, 0.9996528456270659)
    assert np.sum(affected_pixels) == 95

    # Case 3
    results = {'50': {'residuals': np.full(residuals.shape, np.nan)}}
    mean_mf, median_mf, affected_pixels = get_match_fraction_for_position(
        position=(26, 16),
        hypothesis=int(n_frames / 2),
        results=results,
        parang=parang,
        psf_template=psf_template,
        signal_times=signal_times,
        frame_size=frame_size,
    )
    assert mean_mf == 0
    assert median_mf == 0


def test__get_all_match_fraction() -> None:

    # Create fake PSF template
    x, y = np.meshgrid(np.arange(33), np.arange(33))
    model = models.Gaussian2D(x_mean=17, y_mean=17, x_stddev=2, y_stddev=2)
    psf_template = model(x, y)
    psf_template[psf_template < 0.1] = 0

    # Define variables
    n_frames, x_size, y_size = (100, 33, 33)
    frame_size = (x_size, y_size)
    parang = np.linspace(-45, 45, n_frames)

    # Create fake minimal results
    residuals = add_fake_planet(
        stack=np.zeros((n_frames, x_size, y_size)),
        parang=parang,
        psf_template=psf_template,
        polar_position=(Quantity(10, 'pixel'), Quantity(0, 'degree')),
        magnitude=1,
        extra_scaling=1,
        dit_stack=1,
        dit_psf_template=1,
        return_planet_positions=False,
    )
    results = {'50': {'residuals': residuals}}

    # Define hypotheses
    hypotheses = np.full(frame_size, np.nan)
    hypotheses[27, 16] = int(3 * n_frames / 4)
    hypotheses[26, 16] = int(n_frames / 2)

    # Case 1
    mean_mf, median_mf, affected_pixels = get_all_match_fractions(
        results=results,
        roi_mask=np.full(frame_size, True),
        hypotheses=hypotheses,
        parang=parang,
        psf_template=psf_template,
        frame_size=frame_size,
        n_roi_splits=1,
        roi_split=0,
    )
    assert np.isclose(mean_mf[27, 16], 0.32950314505140343)
    assert np.isclose(median_mf[27, 16], 0.32750388582106094)
    assert np.isclose(mean_mf[26, 16], 0.9996813941131175)
    assert np.isclose(median_mf[26, 16], 0.9996528456270659)
