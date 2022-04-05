"""
Methods related to using units and quantities from ``astropy.units``.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from contextlib import nullcontext
from typing import Any, Union

from astropy import units

import numpy as np


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class InstrumentUnitsContext:
    r"""
    Context manager that allows to provide local, instrument-specific
    values for the ``pixscale`` and ``lambda_over_d`` to the registry
    of ``astropy.units``.

    In order to convert between pixels and arcseconds, one has to
    define a pixel scale; this value, however, is instrument-specific.
    Similarly, the value of $\lambda / D$ obviously depends both on the
    wavelength $\lambda$ of the observation and the diameter $D$ of the
    primary mirror.

    If we want to use $\lambda / D$ as a unit (as it is a characteristic
    scale, e.g., for the PSF size), we need to define its value in terms
    of other units somewhere. This class provides a context manager that
    is initialized with values for the pixel scale and $\lambda / D$,
    and it then  provides a (re-usable!) context inside of which an
    :py:class:`astropy.units.Quantity` can be converted freely between
    units of pixels, arc seconds and $\lambda / D$.

    Args:
        pixscale: An :py:class:`astropy.units.Quantity` with units
            `arcsec / pixel` that defines the pixel scale.

            .. admonition:: Example

                For VLT/NACO, the pixscale is typically:
                ``PIXSCALE = 0.0271 arcsec / pixel``.

        lambda_over_d: An :py:class:`astropy.units.Quantity` with units
            `arc seconds` that defines the ratio between the wavelength
            $\lambda$ of the observation and $D$, the diameter of the
            primary mirror.

            .. admonition:: Example

                For *L'*-band data ($\lambda$ = 3800 nm) at the VLT
                ($D$ = 8.2 m), this value is:
                ``lambda_over_d = 0.0956 arcsec``.
    """

    @units.quantity_input(
        pixscale=units.Unit('arcsec / pixel'),
        lambda_over_d=units.Unit('arcsec'),
    )
    def __init__(
        self,
        pixscale: units.Quantity,
        lambda_over_d: units.Quantity,
    ) -> None:

        # Store the values of the pixel scale and lambda over D
        self.pixel_scale = units.pixel_scale(pixscale)
        self.lod_unit = units.def_unit(
            s=['lod', 'lambda_over_d'], represents=lambda_over_d
        )

        # Initialize contexts for units and equivalencies (as empty contexts)
        self.context_units = nullcontext()
        self.context_equivalencies = nullcontext()

    def __enter__(self) -> None:

        # (Re)-create contexts both for the unit (i.e., lambda_over_d) and the
        # equivalency (i.e., pixel and arcseconds).
        # We cannot create these contexts in the constructor, because they are
        # not re-usable, meaning we cannot simply re-enter them after having
        # used their __exit__() method. By re-creating these contexts everytime
        # we call __enter__() on the InstrumentUnitsContext, we ensure that the
        # latter is indeed re-usable.
        self.context_units = units.add_enabled_units(self.lod_unit)
        self.context_equivalencies = units.add_enabled_equivalencies(
            self.pixel_scale
        )

        # Enter the unit conversion contexts we have just created
        self.context_equivalencies.__enter__()
        self.context_units.__enter__()

    def __exit__(self, *exc_details: Any) -> None:

        # Exit the unit conversion context in the order in which we entered
        self.context_units.__exit__(*exc_details)
        self.context_equivalencies.__exit__(*exc_details)


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def flux_ratio_to_magnitudes(
    flux_ratio: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Convert a given flux ratio to magnitudes.

    Example:

        >>> flux_ratio_to_magnitudes(1e-5)
        12.5

    Args:
        flux_ratio: The brightness (or contrast) as a flux ratio;
            either as a single float or as a numpy array of floats.

    Returns:
        The brightness (or contrast) in magnitudes.
    """

    if isinstance(flux_ratio, np.ndarray):
        return np.asarray(-2.5 * np.log10(flux_ratio))
    return -2.5 * float(np.log10(flux_ratio))


def magnitude_to_flux_ratio(
    magnitudes: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Convert magnitudes to a flux ratio.

    Example:
        >>> magnitude_to_flux_ratio(5)
        0.01

    Args:
        magnitudes: The brightness (or contrast) in magnitudes; either
            as a single float or as a numpy array of floats.

    Returns:
        The brightness (or contrast) as a flux ratio.
    """

    return 10 ** (-magnitudes / 2.5)
