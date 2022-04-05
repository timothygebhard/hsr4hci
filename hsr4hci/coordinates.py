"""
Methods for dealing with coordinates.
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

def get_center(frame_size: Tuple[int, int]) -> Tuple[float, float]:
    """
    Using the frame size of an image, determine the precise position
    of its center in the usual coordinate system that we place on our
    data: The *center* of the pixel in the bottom left corner of the
    image is defined as (0, 0), so the bottom left corner of the
    image is located at (-0.5, -0.5).

    This function is essentially a simplified port of the corresponding
    PynPoint function :py:func:`pynpoint.util.image.center_subpixel()`.

    Args:
        frame_size: A tuple of integers `(x_size, y_size)` specifying
            the size of the images (in pixels).

    Returns:
        A tuple of floats, `(center_x, center_y)` containing the
        position of the center of the image.
    """

    return frame_size[0] / 2 - 0.5, frame_size[1] / 2 - 0.5


def polar2cartesian(
    separation: Quantity,
    angle: Quantity,
    frame_size: Tuple[int, int],
) -> Tuple[float, float]:
    """
    Convert a position in astronomical polar coordinates to Cartesian
    coordinates (in pixels).

    Args:
        separation: Separation from the center (as a ``Quantity`` object
            that can be converted to pixels).
        angle: Angle, measured from the up = North direction (this
            corresponds to a -90° offset compared to "normal" polar
            coordinates), as a ``Quantity`` object that can be converted
            to degrees or radian.
        frame_size: A 2-tuple `(x_size, y_size)` of integers specifying
            the size of the frame that we are working with.

    Returns:
        A 2-tuple `(x, y)` containing the Cartesian representation of
        the position specified by the `(separation, angle)` tuple.
        The Cartesian representation uses the astropy convention for the
        position of the origin, and the numpy convention for the order
        of the dimensions.
    """

    # Convert separation and angle to pixel / radian and get raw values
    rho = separation.to('pixel').value
    phi = angle.to('radian').value

    # Convert from astronomical to mathematical polar coordinates
    phi += math.pi / 2

    # Get coordinates of image center
    center = get_center(frame_size=frame_size)

    # Convert from polar to Cartesian coordinates
    x = center[0] + rho * math.cos(phi)
    y = center[1] + rho * math.sin(phi)

    return x, y


def cartesian2polar(
    position: Tuple[float, float],
    frame_size: Tuple[int, int],
) -> Tuple[Quantity, Quantity]:
    """
    Convert a position in Cartesian coordinates (in pixels) to
    astronomical polar coordinates.

    Args:
        position: A tuple `(x, y)` containing a position in Cartesian
            coordinates.
        frame_size: A 2-tuple `(width, height)` of integers specifying
            the size of the frame that we are working with.

    Returns:
        A 2-tuple `(separation, angle)`, where the `separation` is a
        ``Quantity`` in pixels, and `angle` is a ``Quantity`` in radian.
        The `angle` uses the astronomical convention for polar
        coordinates, that is, 0 is "up", not "right" (unlike in
        mathematical polar coordinates).
    """

    # Get coordinates of image center
    center = get_center(frame_size=frame_size)

    # Compute separation and the angle
    separation = math.sqrt(
        (position[0] - center[0]) ** 2 + (position[1] - center[1]) ** 2
    )
    angle = math.atan2(position[1] - center[1], position[0] - center[0])

    # Convert from mathematical to astronomical polar coordinates; constrain
    # the polar angle to the interval [0, 2pi].
    angle -= math.pi / 2
    angle = math.fmod(angle, 2 * math.pi)

    return Quantity(separation, 'pixel'), Quantity(angle, 'radian')
