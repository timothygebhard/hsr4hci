"""
Utility functions related to using units and quantities (astropy.units)
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from astropy import units


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

@units.quantity_input(pixscale=units.Unit('arcsec / pixel'),
                      lambda_over_d=units.Unit('arcsec'))
def set_units_for_instrument(pixscale: units.Quantity,
                             lambda_over_d: units.Quantity):
    """
    Define instrument-specific units and conversions.

    This function makes the pixscale available as a global conversion
    factor, allowing to convert between pixels and arc seconds. It also
    introduces a new unit, lambda_over_d, which is actually equivalent
    to arc seconds up to a constant factor determined by the instrument
    geometry and the wavelength of the filter that is being used.

    After calling this function, we should be able to convert freely
    between units of pixels, arcseconds and lambda_over_d, always using
    the correct instrument-specific conversion factors.

    Args:
        pixscale: The pixel scale of the instrument as an
            astropy.units.Quantity in units of arc seconds per pixel.
        lambda_over_d: The instrument constant lambda over D, as an
            astropy.units.Quantity in units of arc seconds.
    """

    # Construct a new Unit for lambda_over_d and add it to the unit registry.
    lod_unit = units.def_unit(s=['lod', 'lambda_over_d'],
                              represents=lambda_over_d.value * units.arcsec)
    units.add_enabled_units(units=lod_unit)

    # Construct an Equivalency object for the pixel scale and add it to the
    # global unit registry. This object will be used by astropy.units to
    # convert between pixels and arc seconds.
    pixel_scale = units.pixel_scale(pixscale=pixscale)
    units.add_enabled_equivalencies(equivalencies=pixel_scale)

    # This line seems to be necessary to make our units and equivalencies
    # available also outside of the scope of this function
    units.set_enabled_equivalencies(equivalencies=[])


# -----------------------------------------------------------------------------
# TESTING ZONE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    print('\nTEST INSTRUMENT-SPECIFIC UNIT CONVERSIONS\n')

    # Set instrument constants. Usually, this is part of the config file.
    pixscale = units.Quantity(0.0271, 'arcsec / pixel')
    lambda_over_d = units.Quantity(0.096, 'arcsec')

    # Activate the instrument context
    set_units_for_instrument(pixscale=pixscale,
                             lambda_over_d=lambda_over_d)

    # Test if the conversions work correctly
    quantity_1 = units.Quantity(1, 'pixel')
    print(quantity_1,
          '==', quantity_1.to('arcsec'),
          '==', quantity_1.to('lambda_over_d'))
    quantity_2 = units.Quantity(1, 'arcsec')
    print(quantity_2,
          '==', quantity_2.to('pixel'),
          '==', quantity_2.to('lambda_over_d'))
    quantity_3 = units.Quantity(1, 'lambda_over_d')
    print(quantity_3,
          '==', quantity_3.to('arcsec'),
          '==', quantity_3.to('pixel'))
