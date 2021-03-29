"""
Tests for units.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from astropy import units

import pytest

from hsr4hci.units import convert_to_quantity, set_units_for_instrument


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__set_units_for_instrument() -> None:

    # Define pixscale and lambda over D, and set up units for instrument
    pixscale = units.Quantity(0.0271, 'arcsec / pixel')
    lambda_over_d = units.Quantity(0.096, 'arcsec')
    set_units_for_instrument(pixscale=pixscale, lambda_over_d=lambda_over_d)

    # Test case 1: Conversion from pixel to arcsec / lambda_over_d
    quantity = units.Quantity(1.0, 'pixel')
    assert quantity.to('arcsec').value == 0.0271
    assert quantity.to('lambda_over_d').value == 0.28229166666666666

    # Test case 2: Conversion from arcsec to pixel / lambda_over_d
    quantity = units.Quantity(1.0, 'arcsec')
    assert quantity.to('pixel').value == 36.90036900369004
    assert quantity.to('lambda_over_d').value == 10.416666666666666

    # Test case 3: Conversion from lambda_over_d to arcsec / pixel
    quantity = units.Quantity(1.0, 'lambda_over_d')
    assert quantity.to('arcsec').value == 0.096
    assert quantity.to('pixel').value == 3.5424354243542435

    # Test case 4: Overwriting units after they have been set already: This
    # currently does not work, so we need to catch the expected error...
    pixscale = units.Quantity(0.012255, 'arcsec / pixel')
    lambda_over_d = units.Quantity(0.0531, 'arcsec')
    with pytest.raises(RuntimeError) as error:
        set_units_for_instrument(
            pixscale=pixscale,
            lambda_over_d=lambda_over_d
        )
    assert 'Overwriting units is currently not possible!' in str(error)


def test__convert_to_quantity() -> None:

    original_nested_dict = dict(a=dict(b=[42, 'meter']))
    converted_nested_dict = convert_to_quantity(
        config=original_nested_dict, key_tuple=('a', 'b')
    )

    assert converted_nested_dict['a']['b'] == units.Quantity(42, 'meter')
