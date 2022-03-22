"""
Tests for units.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from astropy.units import Quantity, UnitsError, UnitConversionError

import numpy as np
import pytest

from hsr4hci.units import (
    flux_ratio_to_magnitudes,
    InstrumentUnitsContext,
    magnitude_to_flux_ratio,
)


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__instrument_units_context() -> None:
    """
    Test `hsr4hci.units.InstrumentUnitsContext`.
    """

    # Case 1 (illegal constructor argument: pixscale)
    with pytest.raises(UnitsError) as units_error:
        InstrumentUnitsContext(
            pixscale=Quantity(0.0271, 'arcsec'),
            lambda_over_d=Quantity(0.096, 'arcsec'),
        )
    assert "Argument 'pixscale' to function" in str(units_error)

    # Case 2 (illegal constructor argument: lambda_over_d)
    with pytest.raises(UnitsError) as units_error:
        InstrumentUnitsContext(
            pixscale=Quantity(0.0271, 'arcsec / pixel'),
            lambda_over_d=Quantity(0.096, 'gram'),
        )
    assert "Argument 'lambda_over_d' to function" in str(units_error)

    instrument_units_context = InstrumentUnitsContext(
        pixscale=Quantity(0.0271, 'arcsec / pixel'),
        lambda_over_d=Quantity(0.096, 'arcsec'),
    )

    # Case 3 (conversion from pixel to arcsec / lambda_over_d)
    with instrument_units_context:
        quantity = Quantity(1.0, 'pixel')
        assert quantity.to('arcsec').value == 0.0271
        assert quantity.to('lambda_over_d').value == 0.28229166666666666

    # Case 4 (context is re-usable)
    with instrument_units_context:
        quantity = Quantity(1.0, 'pixel')
        assert quantity.to('arcsec').value == 0.0271
        assert quantity.to('lambda_over_d').value == 0.28229166666666666

    # Case 5 (context is local; conversions do not work outside the context)
    with pytest.raises(UnitConversionError) as unit_conversion_error:
        _ = quantity.to('arcsec').value
    assert "'pix' and 'arcsec' (angle) are not" in str(unit_conversion_error)

    # Case 6 (conversion from arcsec to pixel / lambda_over_d)
    with instrument_units_context:
        quantity = Quantity(1.0, 'arcsec')
        assert quantity.to('pixel').value == 36.90036900369004
        assert quantity.to('lambda_over_d').value == 10.416666666666666

    # Case 7 (conversion from lambda_over_d to arcsec / pixel)
    with instrument_units_context:
        quantity = Quantity(1.0, 'lambda_over_d')
        assert quantity.to('arcsec').value == 0.096
        assert quantity.to('pixel').value == 3.5424354243542435

    # Case 8 (contexts can be overwritten / re-defined)
    instrument_units_context = InstrumentUnitsContext(
        pixscale=Quantity(0.271, 'arcsec / pixel'),
        lambda_over_d=Quantity(0.96, 'arcsec'),
    )
    with instrument_units_context:
        quantity = Quantity(1.0, 'pixel')
        assert quantity.to('arcsec').value == 0.271
        assert quantity.to('lambda_over_d').value == 0.2822916666666667

    # Case 9 (different contexts can co-exist)
    context_a = InstrumentUnitsContext(
        pixscale=Quantity(0.0271, 'arcsec / pixel'),
        lambda_over_d=Quantity(0.096, 'arcsec'),
    )
    context_b = InstrumentUnitsContext(
        pixscale=Quantity(0.271, 'arcsec / pixel'),
        lambda_over_d=Quantity(0.96, 'arcsec'),
    )
    quantity = Quantity(1.0, 'pixel')
    with context_a:
        assert quantity.to('arcsec').value == 0.0271
    with context_b:
        assert quantity.to('arcsec').value == 0.271


def test__flux_ratio_to_magnitudes() -> None:
    """
    Test `hsr4hci.units.flux_ratio_to_magnitudes`.
    """

    assert flux_ratio_to_magnitudes(100) == -5
    assert np.allclose(
        flux_ratio_to_magnitudes(np.array([100, 0.01])), np.array([-5, 5])
    )


def test__magnitude_to_flux_ratio() -> None:
    """
    Test `hsr4hci.units.magnitude_to_flux_ratio`.
    """

    assert magnitude_to_flux_ratio(-5) == 100
    assert np.allclose(
        magnitude_to_flux_ratio(np.array([-5, 5])), np.array([100, 0.01])
    )
