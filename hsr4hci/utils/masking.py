"""
Utility functions for creating and working with (binary) masks.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import List, Optional, Tuple

from astropy.units import Quantity
from skimage.morphology import binary_dilation, disk

import numpy as np

from hsr4hci.utils.coordinates import (
    get_center,
    cartesian2polar,
    polar2cartesian
)
from hsr4hci.utils.signal_masking import get_signal_length


# -----------------------------------------------------------------------------
# BASE MASKS (INPUT PARAMETERS IN PIXELS)
# -----------------------------------------------------------------------------

def get_circle_mask(
    mask_size: Tuple[int, int],
    radius: float,
    center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Create a circle mask.

    Note: This function uses the numpy convention for coordinates!

    Args:
        mask_size: A tuple `(x_size, y_size)` containing the size of the
            mask (in pixels) to be created.
        radius: Radius of the disk (in pixels).
        center: A tuple `(x, y)` containing the center of the circle.
            If None is given, the circle will be centered within the
            mask (this is the default).

    Returns:
        A numpy array of the given `mask_size` which is False
        everywhere, except in a circular region of given radius around
        the specified `center`.
    """

    x, y = np.ogrid[:mask_size[0], :mask_size[1]]

    if center is None:
        center = get_center(mask_size)

    return np.asarray(
        (x - center[0]) ** 2 + (y - center[1]) ** 2 < radius ** 2
    )


def get_annulus_mask(
    mask_size: Tuple[int, int],
    inner_radius: float,
    outer_radius: float,
    center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Create an annulus-shaped mask.

    Note: This function uses the numpy convention for coordinates!

    Args:
        mask_size: A tuple (width, height) containing the size of the
            mask (in pixels) to be created. Should match the size of
            the array which is masked.
        inner_radius: Inner radius (in pixels) of the annulus mask.
        outer_radius: Outer radius (in pixels) of the annulus mask.
        center: A tuple `(x, y)` containing the center of the annulus.
            If None is given, the annulus will be centered within the
            mask (this is the default).

    Returns:
        A 2D numpy array of size `mask_size` which masks an annulus
        with a given `inner_radius` and `outer_radius`.
    """

    return np.asarray(
        np.logical_xor(
            get_circle_mask(mask_size, inner_radius, center),
            get_circle_mask(mask_size, outer_radius, center),
        )
    )


# -----------------------------------------------------------------------------
# DERIVED MASKS (INPUT PARAMETERS IN PHYSICAL UNITS)
# -----------------------------------------------------------------------------

def get_roi_mask(
    mask_size: Tuple[int, int],
    inner_radius: Quantity,
    outer_radius: Quantity,
) -> np.ndarray:
    """
    Get a numpy array masking the pixels within the region of interest.

    Args:
        mask_size: A tuple (width, height) containing the spatial size
            of the input stack.
        inner_radius: Inner radius of the region of interest (as an
            astropy.units.Quantity that can be converted to pixels).
        outer_radius: Outer radius of the region of interest (as an
            astropy.units.Quantity that can be converted to pixels).

    Returns:
        A 2D numpy array of size `mask_size` which masks the pixels
        within the specified region of interest.
    """

    return get_annulus_mask(
        mask_size=mask_size,
        inner_radius=inner_radius.to('pixel').value,
        outer_radius=outer_radius.to('pixel').value,
    )


def get_predictor_mask(
    mask_size: Tuple[int, int],
    position: Tuple[int, int],
    annulus_width: Quantity,
    radius_position: Quantity,
    radius_mirror_position: Quantity,
) -> np.ndarray:
    """
    Create a mask that selects the potential predictors for a position.

    For a given position (x, y), this mask selects all pixels in a
    circular region with radius `circular_radius` around the position,
    all pixels on an annulus at the same separation as (x, y) of width
    `annulus_width`, and finally yet another circular region with radius
    `circular_radius` centered on (-x, -y), where the center of the
    frame is taken as the origin of the coordinate system.

    If we remove an exclusion region from this region, we get the
    selection mask, that is, the mask that actually selects the pixels
    to be used as predictors for a given position.

    Args:
        mask_size: A tuple (width, height) that specifies the size of
            the mask to be created in pixels.
        position: A tuple (x, y) specifying the position for which this
            mask is created, i.e., the mask selects the pixels that are
            used as predictors for (x, y).
        annulus_width: The width (as an astropy.units.Quantity that can
            be converted to pixels) of the annulus used to select
            potential predictor pixels.
        radius_position: The radius (as an astropy.units.Quantity that
            can be converted to pixels) of the circular region around
            the `position` that is used to select potential predictors.
        radius_mirror_position: The radius (as an astropy.units.Quantity
            that can be converted to pixels) of the circular region
            around the mirrored `position` that is used to select
            potential predictors.

    Returns:
        A 2D numpy array containing a mask that contains all potential
        predictors for the pixel at the given `position`, that is,
        including the pixels that we must not use because they are
        not causally disconnected
    """

    # Compute mask center and separation of `position` from the center
    center = get_center(mask_size)
    separation = np.hypot((position[0] - center[0]), (position[1] - center[1]))

    # Initialize an empty mask of the desired size
    mask = np.full(mask_size, False)

    # Add circular selection mask at position (x, y)
    circular_mask = get_circle_mask(
        mask_size=mask_size,
        radius=radius_position.to('pixel').value,
        center=position,
    )
    mask = np.logical_or(mask, circular_mask)

    # Add circular selection mask at mirror position (-x, -y)
    mirror_position = (
        2 * center[0] - position[0],
        2 * center[1] - position[1],
    )
    circular_mask = get_circle_mask(
        mask_size=mask_size,
        radius=radius_mirror_position.to('pixel').value,
        center=mirror_position,
    )
    mask = np.logical_or(mask, circular_mask)

    # Add annulus-shaped selection mask of given width at the given separation
    half_width = annulus_width.to('pixel').value / 2
    annulus = get_annulus_mask(
        mask_size=mask_size,
        inner_radius=(separation - half_width),
        outer_radius=(separation + half_width),
    )
    mask = np.logical_or(mask, annulus)

    return mask


def get_exclusion_mask(
    mask_size: Tuple[int, int],
    position: Tuple[float, float],
    parang: np.ndarray,
    signal_time: Optional[int],
    psf_template: np.ndarray,
    psf_radius: float,
    dilation_size: Optional[int] = None,
    use_field_rotation: bool = True,
) -> np.ndarray:
    """
    Get a mask of the pixels that we must not use as predictors.

    Args:
        mask_size:
        position:
        parang:
        signal_time:
        psf_template:
        psf_radius:
        dilation_size:
        use_field_rotation:

    Returns:

    """

    # Make sure that the options do not contradict each other
    if signal_time is None and use_field_rotation:
        raise ValueError(
            'No signal_time given; this only works in combination with '
            'use_field_rotation=False!'
        )

    # Defines shortcuts
    n_frames = len(parang)

    # Get polar representation of the current position
    rho, phi = cartesian2polar(position, mask_size)

    # In case we use the field rotation (e.g., for signal masking models), we
    # need to compute the size of it based on the expected signal length at
    # the current position. Basically, we have two exclude all pixels whose
    # time series have bumps that (temporally) overlap with the (potential)
    # signal bump in the pixel at the current `position`.
    if use_field_rotation and signal_time is not None:

        if signal_time < 0:
            raise ValueError('Negative signal times are not allowed!')

        # Get expected total signal length at the current (spatio-temporal)
        # position; this is a quantity in units of "number of frames"
        length_1, length_2 = get_signal_length(
            position=position,
            signal_time=signal_time,
            parang=parang,
            frame_size=mask_size,
            psf_template=psf_template,
        )
        signal_length = int(length_1 + length_2)

        # Compute the exclusion angle in forward and backward direction
        lower = max(0, signal_time - signal_length)
        upper = min(n_frames - 1, signal_time + signal_length)
        angle_1 = parang[signal_time] - parang[lower]
        angle_2 = parang[signal_time] - parang[upper]

        angles = np.linspace(
            phi.to('degree').value + angle_1,
            phi.to('degree').value + angle_2,
            n_frames
        )

    # In case we do not use the field rotation to compute the exclusion region,
    # as it is the case for the "baseline" models, we simply set the exclusion
    # angles to zero; this will result in a circular exclusion mask
    else:
        angles = np.array([phi.to('degree').value])

    # Initialize the exclusion mask
    exclusion_mask = np.zeros(mask_size)

    # Compute an arc
    for angle in angles:
        x, y = polar2cartesian(
            separation=rho,
            angle=Quantity(angle, 'degree'),
            frame_size=mask_size
        )
        exclusion_mask[int(x), int(y)] = 1

    if dilation_size is not None:
        selem = disk(radius=int(np.ceil(2 * psf_radius + dilation_size)))
    else:
        selem = disk(radius=int(np.ceil(2 * psf_radius)))

    exclusion_mask = binary_dilation(image=exclusion_mask, selem=selem)

    return exclusion_mask


def get_selection_mask(
    mask_size: Tuple[int, int],
    position: Tuple[int, int],
    signal_time: Optional[int],
    parang: np.ndarray,
    annulus_width: Quantity,
    radius_position: Quantity,
    radius_mirror_position: Quantity,
    psf_template: np.ndarray,
    psf_radius: float,
    dilation_size: int,
    use_field_rotation: bool = True,
) -> np.ndarray:
    """
    Get the mask that selects the predictor pixels for a given position.

    Args:
        mask_size: A tuple (width, height) that specifies the size of
            the mask to be created in pixels.
        position: A tuple (x, y) specifying the position for which this
            mask is created, i.e., the mask selects the pixels that are
            used as predictors for (x, y).
        signal_time: FIXME
        parang: FIXME
        annulus_width: The width (as an astropy.units.Quantity that can
            be converted to pixels) of the annulus used in the
            get_predictor_mask() function.
        radius_position: The radius (as an astropy.units.Quantity that
            can be converted to pixels) of the circular region around
            the `position` (and the mirrored position) that is used in
            the get_predictor_mask() function.
        radius_mirror_position: The radius (as an astropy.units.Quantity
            that can be converted to pixels) of the circular region
            around the mirrored `position`.
        psf_template:  FIXME
        psf_radius: FIXME
        dilation_size: FIXME
        use_field_rotation: A boolean indicating whether or not to use
            the field rotation when determining the exclusion region.

    Returns:
        A 2D numpy array containing a mask that selects the pixels to
        be used as predictors for the pixel at the given `position`.
    """

    # Get the mask that selects all potential predictor pixels
    predictor_mask = get_predictor_mask(
        mask_size=mask_size,
        position=position,
        annulus_width=annulus_width,
        radius_position=radius_position,
        radius_mirror_position=radius_mirror_position,
    )

    # Get exclusion mask (i.e., pixels we must not use as predictors)
    exclusion_mask = get_exclusion_mask(
        mask_size=mask_size,
        position=position,
        parang=parang,
        signal_time=signal_time,
        psf_template=psf_template,
        psf_radius=psf_radius,
        dilation_size=dilation_size,
        use_field_rotation=use_field_rotation,
    )

    # Create the actual selection mask by removing the exclusion mask
    # from the predictor mask
    selection_mask = np.logical_and(
        np.logical_not(exclusion_mask), predictor_mask
    )

    return np.asarray(selection_mask)


# -----------------------------------------------------------------------------
# OTHER FUNCTIONS
# -----------------------------------------------------------------------------

def get_positions_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Convert a numpy mask into a list of positions selected by that mask.

    Args:
        mask: A numpy array containing only boolean values (or values
            that can be interpreted as such).

    Returns:
        A sorted list of all positions (x, y) with mask[x, y] == True.
    """

    return sorted(list((x, y) for x, y in zip(*np.where(mask))))
