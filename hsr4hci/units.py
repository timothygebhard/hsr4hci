"""
Utility functions related to using units and quantities (astropy.units)
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Dict, Tuple

from astropy import units

from hsr4hci.general import get_from_nested_dict, set_in_nested_dict


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def convert_to_quantity(
    config: Dict[str, Any],
    key_tuple: Tuple[str, ...],
) -> Dict[str, Any]:
    """
    Auxiliary function to convert a entry in a given configuration
    dictionary to an `astropy.units.Quantity` object.

    Args:
        config: The (possibly nested) dictionary containing the raw
            configuration.
        key_tuple: The path (within the nested dictionary) to the entry
            that we want to convert to a astro.units.Quantity. This
            entry is assumed to be a Sequence of the for (value, unit),
            where value is an integer or a float, and unit is a string
            specifying the unit. Example: (0.5, "arcsec").

    Returns:
        The original `config` dictionary, with the entry specified by
        the `key_tuple` converted to an `astropy.units.Quantity` object.
    """

    # Get raw value from nested dictionary with the configuration
    value = get_from_nested_dict(nested_dict=config, location=key_tuple)

    # Write the converted value back to the configuration dictionary
    set_in_nested_dict(
        nested_dict=config, location=key_tuple, value=units.Quantity(*value)
    )

    return config


@units.quantity_input(
    pixscale=units.Unit('arcsec / pixel'), lambda_over_d=units.Unit('arcsec')
)
def set_units_for_instrument(
    pixscale: units.Quantity,
    lambda_over_d: units.Quantity,
    verbose: bool = True,
) -> None:
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
        verbose: Whether or not to print a message about the instrument-
            specific unit conversion context that was activated by
            calling this function.
    """

    # Construct a new Unit for lambda_over_d and add it to the unit registry.
    lod_unit = units.def_unit(
        s=['lod', 'lambda_over_d'],
        represents=lambda_over_d.value * units.arcsec,
    )
    try:
        units.add_enabled_units(units=lod_unit)
    except ValueError:
        raise RuntimeError('Overwriting units is currently not possible!')

    # Construct an Equivalency object for the pixel scale and add it to the
    # global unit registry. This object will be used by astropy.units to
    # convert between pixels and arc seconds.
    pixel_scale = units.pixel_scale(pixscale=pixscale)
    units.add_enabled_equivalencies(equivalencies=pixel_scale)

    # This line seems to be necessary to make our units and equivalencies
    # available also outside of the scope of this function
    units.set_enabled_equivalencies(equivalencies=[])

    if verbose:
        print('Activated instrument-specific unit conversion context:')
        print('  PIXSCALE      =', pixscale)
        print('  LAMBDA_OVER_D =', lambda_over_d, '\n')


def to_pixel(quantity: units.Quantity) -> float:
    """
    Convert a given quantity to pixels and return the value as a float.

    Args:
        quantity: A compatible astropy.units.Quantity, that is, a
        quantity that can be converted to pixel.

    Returns:
        The value of `quantity` in pixels as a simple float.
    """

    return float(quantity.to('pixel').value)
