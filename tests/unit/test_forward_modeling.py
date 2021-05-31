"""
Tests for forward_modeling.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from astropy.modeling import models
from astropy.units import Quantity

import numpy as np

from hsr4hci.coordinates import cartesian2polar
from hsr4hci.forward_modeling import (
    add_fake_planet,
    get_time_series_for_position,
    get_time_series_for_position__full_stack,
)


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__add_fake_planet() -> None:

    x, y = np.meshgrid(np.arange(33), np.arange(33))
    gaussian = models.Gaussian2D(x_mean=16, y_mean=16)
    psf_template = np.asarray(gaussian(x, y))

    n_frames = 91
    frame_size = (51, 51)

    # Case 1
    stack_with_planet, planet_positions = add_fake_planet(
        stack=np.zeros((n_frames,) + frame_size),
        parang=np.linspace(0, 90, n_frames),
        psf_template=psf_template,
        polar_position=(Quantity(10, 'pixel'), Quantity(0, 'degree')),
        magnitude=1,
        extra_scaling=1,
        dit_stack=1,
        dit_psf_template=1,
        return_planet_positions=True,
        interpolation='bilinear',
    )
    separations = [
        cartesian2polar(_, frame_size)[0].to('pixel').value
        for _ in planet_positions
    ]
    assert all(np.isclose(_, 10) for _ in separations)
    angles = [
        cartesian2polar(_, frame_size)[1].to('degree').value
        for _ in planet_positions
    ]
    assert np.allclose(np.around(angles, 3), np.linspace(0, -90, n_frames))


def test__get_time_series_for_position() -> None:

    x, y = np.meshgrid(np.arange(33), np.arange(33))
    gaussian = models.Gaussian2D(x_mean=16, y_mean=16)
    psf_template = np.asarray(gaussian(x, y))

    n_frames = 91
    frame_size = (51, 51)
    signal_time = 30

    time_series = get_time_series_for_position(
        position=(11, 13),
        signal_time=signal_time,
        frame_size=frame_size,
        parang=np.linspace(10, 50, n_frames),
        psf_template=psf_template,
    )

    assert np.max(time_series) == 1
    assert np.argmax(time_series) == signal_time


def test__get_time_series_for_position__full_stack() -> None:

    x, y = np.meshgrid(np.arange(33), np.arange(33))
    gaussian = models.Gaussian2D(x_mean=16, y_mean=16)
    psf_template = np.asarray(gaussian(x, y))

    n_frames = 91
    frame_size = (51, 51)
    signal_time = 60

    time_series = get_time_series_for_position__full_stack(
        position=(32, 13),
        signal_time=signal_time,
        frame_size=frame_size,
        parang=np.linspace(10, 50, n_frames),
        psf_template=psf_template,
    )

    assert np.max(time_series) == 1
    assert np.argmax(time_series) == signal_time
