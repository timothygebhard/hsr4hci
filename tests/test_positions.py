"""
Tests for positions.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from astropy.units import Quantity

import pytest

from hsr4hci.positions import (
    get_injection_position,
    get_reference_positions,
    rotate_reference_positions,
)


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__get_injection_position() -> None:

    # Case 1
    with pytest.raises(ValueError) as value_error:
        get_injection_position(
            separation=Quantity(3, 'pixel'), azimuthal_position='g'
        )
    assert 'azimuthal_position must be' in str(value_error)

    # Case 2
    injection_position = get_injection_position(
        separation=Quantity(3, 'pixel'), azimuthal_position='a'
    )
    assert injection_position[0] == Quantity(3, 'pixel')
    assert injection_position[1] == Quantity(270, 'degree')

    # Case 3
    injection_position = get_injection_position(
        separation=Quantity(0.5, 'arcsec'), azimuthal_position='f'
    )
    assert injection_position[0] == Quantity(0.5, 'arcsec')
    assert injection_position[1] == Quantity(210, 'degree')


def test__get_reference_positions() -> None:

    # Case 1
    reference_positions = get_reference_positions(
        polar_position=(Quantity(5, 'pixel'), Quantity(0, 'degree')),
        aperture_radius=Quantity(2.5, 'pixel'),
        exclusion_angle=Quantity(60, 'degree'),
    )
    assert len(reference_positions) == 5
    assert all(_[0] == Quantity(5, 'pixel') for _ in reference_positions)
    assert reference_positions[2][1] == Quantity(180, 'degree')

    # Case 2
    reference_positions = get_reference_positions(
        polar_position=(Quantity(0.6, 'arcsec'), Quantity(31.4159, 'degree')),
        aperture_radius=Quantity(0.2, 'arcsec'),
        exclusion_angle=None,
    )
    assert len(reference_positions) == 6
    assert all(_[0] == Quantity(0.6, 'arcsec') for _ in reference_positions)

    # Case 2
    reference_positions = get_reference_positions(
        polar_position=(Quantity(8, 'pixel'), Quantity(-5, 'degree')),
        aperture_radius=Quantity(2, 'pixel'),
        exclusion_angle=Quantity(0, 'degree'),
    )
    assert len(reference_positions) == 11


def test__rotate_positions() -> None:

    # Case 1
    reference_positions = get_reference_positions(
        polar_position=(Quantity(5, 'pixel'), Quantity(0, 'degree')),
        aperture_radius=Quantity(2.5, 'pixel'),
        exclusion_angle=Quantity(0, 'degree'),
    )
    rotated_reference_positions = rotate_reference_positions(
        reference_positions=reference_positions, n_steps=5
    )
    assert len(rotated_reference_positions) == 5
    assert rotated_reference_positions[1][0][1] == Quantity(75, 'degree')
    assert rotated_reference_positions[2][1][1] == Quantity(150, 'degree')

    # Case 2
    reference_positions = get_reference_positions(
        polar_position=(Quantity(2, 'pixel'), Quantity(0, 'degree')),
        aperture_radius=Quantity(2, 'pixel'),
        exclusion_angle=Quantity(0, 'degree'),
    )
    assert len(reference_positions) == 1
    with pytest.raises(RuntimeError) as runtime_error:
        rotate_reference_positions(
            reference_positions=reference_positions, n_steps=5
        )
    assert 'Need at least two reference positions to' in str(runtime_error)
