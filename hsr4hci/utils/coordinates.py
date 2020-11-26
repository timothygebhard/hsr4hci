"""
Utility functions for dealing with coordinates.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Tuple

import math

from astropy.units import Quantity


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def polar2cartesian(
    separation: Quantity,
    angle: Quantity,
    frame_size: Tuple[int, int],
) -> Tuple[float, float]:
    """
    Convert a position in astronomical polar coordinates to a Cartesian
    coordinate in pixels.

    Args:
        separation: Separation from the center (as a Quantity object
            that can be converted to pixels).
        angle: Angle, measured from the up = North direction (this
            corresponds to a -90° offset compared to "normal" polar
            coordinates), as a Quantity object that can be converted
            to degrees or radian.
        frame_size: A 2-tuple `(width, height)` of integers specifying
            the size of the frame that we are working with.

    Returns:
        A 2-tuple `(x, y)` containing the Cartesian representation of
        the position specified by the `(separation, angle)` tuple.
        The Cartesian representation uses the astropy-convention for the
        position of the origin, and the numpy convention for the order
        of the dimensions.
    """

    # Convert separation and angle to pixel / radian and get raw values
    rho = separation.to('pixel').value
    phi = angle.to('radian').value

    # Convert from astronomical to mathematical polar coordinates
    phi += math.pi / 2

    # Convert from polar to Cartesian coordinates
    x = frame_size[0] / 2 + rho * math.cos(phi)
    y = frame_size[1] / 2 + rho * math.sin(phi)

    return x, y
