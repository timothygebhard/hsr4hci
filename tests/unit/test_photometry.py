"""
Tests for photometry.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from astropy.modeling import models
from astropy.units import Quantity

import numpy as np
import pytest

# noinspection PyProtectedMember
from hsr4hci.photometry import (
    _gaussian_integral,
    _get_flux__as,
    _get_flux__ass,
    _get_flux__f,
    _get_flux__fs,
    _get_flux__p,
    get_flux,
    get_fluxes_for_polar_positions,
    get_stellar_flux,
)
from hsr4hci.coordinates import polar2cartesian


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test___gaussian_integral() -> None:
    """
    Test for `hsr4hci.photometry._gaussian_integral`.
    """

    # Case 1
    value = _gaussian_integral(amplitude=1, sigma=1, radius=0.5)
    assert np.isclose(value, 2 * np.pi * (1 - np.exp(-1 / 8)))

    # Case 2
    value = _gaussian_integral(amplitude=1, sigma=1, radius=np.infty)
    assert np.isclose(value, 2 * np.pi)

    # Case 3
    value = _gaussian_integral(amplitude=1, sigma=1, radius=0)
    assert np.isclose(value, 0)


def test___get_flux__as() -> None:
    """
    Test for `hsr4hci.photometry._get_flux__as`.
    """

    # Case 1
    frame = np.zeros((10, 10))
    frame[3, 3] = 1
    position, flux = _get_flux__as(
        frame=frame,
        position=(3, 3),
        aperture_radius=Quantity(2, 'pixel'),
    )
    assert position == (3, 3)
    assert flux == 1

    # Case 2
    frame = np.ones((10, 10))
    position, flux = _get_flux__as(
        frame=frame,
        position=(5, 4),
        aperture_radius=Quantity(1, 'pixel'),
    )
    assert position == (5, 4)
    assert np.isclose(flux, np.pi)


def test___get_flux__ass() -> None:
    """
    Test for `hsr4hci.photometry._get_flux__ass`.
    """

    # Case 1
    x, y = np.meshgrid(np.arange(33), np.arange(33))
    gaussian = models.Gaussian2D(
        x_mean=17, x_stddev=1, y_mean=17, y_stddev=1, amplitude=1
    )
    position, flux = _get_flux__ass(
        frame=gaussian(x, y),
        position=(16, 16),
        aperture_radius=Quantity(2, 'pixel'),
        search_radius=Quantity(2, 'pixel'),
    )
    assert position == (17, 17)
    assert np.isclose(flux, 5.27522347777759)


def test___get_flux__p() -> None:
    """
    Test for `hsr4hci.photometry._get_flux__p`.
    """

    # Case 1
    frame = np.zeros((11, 11))
    frame[:, :6] = 1
    position, flux = _get_flux__p(
        frame=frame,
        position=(5, 5),
    )
    assert position == (5, 5)
    assert np.isclose(flux, np.pi / 4)


def test___get_flux__f() -> None:
    """
    Test for `hsr4hci.photometry._get_flux__f`.
    """

    # Case 1
    x, y = np.meshgrid(np.arange(33), np.arange(33))
    gaussian = models.Gaussian2D(
        x_mean=17, x_stddev=1, y_mean=17, y_stddev=1, amplitude=1
    )
    position, flux = _get_flux__f(
        frame=gaussian(x, y),
        position=(17, 17),
    )
    assert position == (17, 17)
    assert np.isclose(flux, 2 * np.pi * (1 - np.exp(-1 / 8)))


def test___get_flux__fs() -> None:
    """
    Test for `hsr4hci.photometry._get_flux__fs`.
    """

    # Case 1
    x, y = np.meshgrid(np.arange(33), np.arange(33))
    gaussian = models.Gaussian2D(
        x_mean=17,
        x_stddev=1,
        y_mean=17,
        y_stddev=1,
        amplitude=1,
    )
    position, flux = _get_flux__fs(
        frame=gaussian(x, y),
        position=(16, 16),
        search_radius=Quantity(2, 'pixel'),
    )
    assert np.allclose(position, (17, 17))
    assert np.isclose(flux, 2 * np.pi * (1 - np.exp(-1 / 8)))

    # Case 2 (ensure that signals "in the distance" do not affect the fit)
    x, y = np.meshgrid(np.arange(33), np.arange(33))
    frame = np.zeros((33, 33))
    frame += models.Gaussian2D(
        x_mean=17,
        x_stddev=1,
        y_mean=17,
        y_stddev=1,
        amplitude=1,
    )(x, y)
    frame += models.Gaussian2D(
        x_mean=24,
        x_stddev=1,
        y_mean=24,
        y_stddev=1,
        amplitude=1e4,
    )(x, y)
    position, flux = _get_flux__fs(
        frame=frame,
        position=(16, 16),
        search_radius=Quantity(2, 'pixel'),
    )
    assert np.allclose(position, (17, 17))
    assert np.isclose(flux, 2 * np.pi * (1 - np.exp(-1 / 8)))


def test__get_flux() -> None:
    """
    Test for `hsr4hci.photometry.get_flux`.
    """

    # Prepare data
    x, y = np.meshgrid(np.arange(33), np.arange(33))
    gaussian = models.Gaussian2D(
        x_mean=17,
        x_stddev=1,
        y_mean=17,
        y_stddev=1,
        amplitude=1,
    )
    frame = gaussian(x, y)
    position = (16, 16)
    aperture_radius = Quantity(1, 'pixel')
    search_radius = Quantity(1, 'pixel')

    # Case 1
    position_1, flux_1 = get_flux(
        frame=frame,
        position=position,
        mode='AS',
        aperture_radius=aperture_radius,
        search_radius=search_radius,
    )
    position_2, flux_2 = _get_flux__as(frame, position, aperture_radius)
    assert position_1 == position_2
    assert np.isclose(flux_1, flux_2)

    # Case 2
    position_1, flux_1 = get_flux(
        frame=frame,
        position=position,
        mode='ASS',
        aperture_radius=aperture_radius,
        search_radius=search_radius,
    )
    position_2, flux_2 = _get_flux__ass(
        frame, position, aperture_radius, search_radius
    )
    assert position_1 == position_2
    assert np.isclose(flux_1, flux_2)

    # Case 3
    position_1, flux_1 = get_flux(
        frame=frame,
        position=position,
        mode='P',
        aperture_radius=aperture_radius,
        search_radius=search_radius,
    )
    position_2, flux_2 = _get_flux__p(frame, position)
    assert position_1 == position_2
    assert np.isclose(flux_1, flux_2)

    # Case 4
    position_1, flux_1 = get_flux(
        frame=frame,
        position=position,
        mode='F',
        aperture_radius=aperture_radius,
        search_radius=search_radius,
    )
    position_2, flux_2 = _get_flux__f(frame, position)
    assert position_1 == position_2
    assert np.isclose(flux_1, flux_2)

    # Case 5
    position_1, flux_1 = get_flux(
        frame=frame,
        position=position,
        mode='FS',
        aperture_radius=aperture_radius,
        search_radius=search_radius,
    )
    position_2, flux_2 = _get_flux__fs(frame, position, search_radius)
    assert position_1 == position_2
    assert np.isclose(flux_1, flux_2)

    # Case 6
    with pytest.raises(ValueError) as error:
        get_flux(
            frame=frame,
            position=position,
            mode='illegal',
            aperture_radius=aperture_radius,
            search_radius=search_radius,
        )
    assert 'Mode "illegal" not supported!' in str(error)


def test__get_stellar_flux() -> None:
    """
    Test for `hsr4hci.photometry.get_stellar_flux`.
    """

    # Case 1
    x, y = np.meshgrid(np.arange(63), np.arange(63))
    gaussian = models.Gaussian2D(
        x_mean=32,
        x_stddev=1,
        y_mean=32,
        y_stddev=1,
        amplitude=1,
    )
    flux = get_stellar_flux(
        psf_template=gaussian(x, y),
        mode='FS',
        search_radius=Quantity(2, 'pixel'),
        dit_stack=1,
        dit_psf_template=0.1,
        scaling_factor=0.02,
    )
    assert np.isclose(flux, (2 * np.pi * (1 - np.exp(-1 / 8))) / 0.1 / 0.02)


def test__get_fluxes_for_polar_positions() -> None:
    """
    Test for `hsr4hci.photometry.get_fluxes_for_polar_positions`.
    """

    frame_size = (101, 101)
    polar_positions = [
        (Quantity(15, 'pixel'), Quantity(0, 'degree')),
        (Quantity(30, 'pixel'), Quantity(120, 'degree')),
        (Quantity(45, 'pixel'), Quantity(240, 'degree')),
    ]

    # Prepare test data
    frame = np.zeros(frame_size)
    x, y = np.meshgrid(np.arange(frame_size[0]), np.arange(frame_size[1]))
    for polar_position in polar_positions:
        x_mean, y_mean = polar2cartesian(
            *polar_position, frame_size=frame_size
        )
        gaussian = models.Gaussian2D(
            x_mean=x_mean,
            x_stddev=1,
            y_mean=y_mean,
            y_stddev=1,
            amplitude=1,
        )
        frame += gaussian(x, y)

    # Case 1
    fluxes = get_fluxes_for_polar_positions(
        polar_positions=polar_positions,
        frame=frame,
        mode='ASS',
        aperture_radius=Quantity(5, 'pixel'),
        search_radius=Quantity(1, 'pixel'),
    )
    assert all(np.isclose(_, 2 * np.pi) for _ in fluxes)
