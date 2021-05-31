"""
Tests for evaluation.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from math import atan2
from astropy.units import Quantity

from hsr4hci.coordinates import get_center, polar2cartesian, cartesian2polar


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__get_center() -> None:

    assert get_center((1, 1)) == (0.0, 0.0)
    assert get_center((2, 2)) == (0.5, 0.5)
    assert get_center((3, 3)) == (1.0, 1.0)


def test__polar2cartesian() -> None:

    cartesian = polar2cartesian(
        separation=Quantity(0, 'pixel'),
        angle=Quantity(90, 'degree'),
        frame_size=(101, 101),
    )
    assert cartesian == (50.0, 50.0)

    cartesian = polar2cartesian(
        separation=Quantity(10, 'pixel'),
        angle=Quantity(0, 'degree'),
        frame_size=(101, 101),
    )
    assert cartesian == (50.0, 60.0)

    cartesian = polar2cartesian(
        separation=Quantity(10, 'pixel'),
        angle=Quantity(90, 'degree'),
        frame_size=(101, 101),
    )
    assert cartesian == (40.0, 50.0)


def test__cartesian2polar() -> None:

    polar = cartesian2polar(
        position=(50, 50),
        frame_size=(101, 101),
    )
    assert polar[0] == Quantity(0, 'pixel')

    polar = cartesian2polar(
        position=(40, 50),
        frame_size=(101, 101),
    )
    assert polar == (Quantity(10, 'pixel'), Quantity(90, 'degree'))

    polar = cartesian2polar(
        position=(53, 54),
        frame_size=(101, 101),
    )
    assert polar == (Quantity(5, 'pixel'), Quantity(-atan2(3, 4), 'radian'))
