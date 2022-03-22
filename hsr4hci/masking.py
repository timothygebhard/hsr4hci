"""
Methods for creating and working with (binary) masks.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import List, Optional, Tuple

from astropy.units import Quantity
from scipy import ndimage

import numpy as np

from hsr4hci.coordinates import get_center


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

    Note: This function uses the *numpy convention* for coordinates!

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

    x, y = np.ogrid[: mask_size[0], : mask_size[1]]

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

    Note: This function uses the *numpy convention* for coordinates!

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

    Note: This function uses the *numpy convention* for coordinates!

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
    radius_position: Quantity,
    radius_opposite: Quantity,
) -> np.ndarray:
    """
    Create a mask that selects the potential predictors for a position.

    For a given position (x, y), this mask selects all pixels in a
    circular region with radius `radius_position` around the position,
    and another circular region with radius `radius_opposite` centered
    on (-x, -y), where the center of the frame is taken as the origin
    of the coordinate system.

    Note: This function uses the *astropy convention* for coordinates!

    Args:
        mask_size: A tuple `(x_size, y_size)` that specifies the size
            of the mask to be created in pixels.
        position: A tuple `(x, y)` specifying the position / pixel for
            which this mask is created. The `position` is specified in
            astropy / matplotlib coordinates, *not* numpy coordinates!
        radius_position: The radius (as an `astropy.units.Quantity` that
            can be converted to pixels) of the circular region around
            the `position` that is used to select potential predictors.
        radius_opposite: The radius (as an `astropy.units.Quantity` that
            can be converted to pixels) of the circular region around
            the opposite `position`, that is, the position that we get
            if we mirror `position` across the center of the frame.

    Returns:
        A 2D numpy array containing a mask that contains all potential
        predictors for the pixel at the given `position`, that is,
        including the pixels that we must not use because they are
        not causally disconnected
    """

    # Add circular selection mask at position (x, y)
    # We need to flip the `position` because get_circle_mask() uses the numpy
    # convention whereas `position` is assumed to be in the astropy convention
    predictor_mask = get_circle_mask(
        mask_size=mask_size,
        radius=radius_position.to('pixel').value,
        center=position[::-1],
    )

    # Add circular selection mask at opposite (= mirror) position (-x, -y) by
    # first creating a circle at the `position` (flip for numpy coordinates)
    # and then rotating the entire mask by 180 degree
    circular_mask = get_circle_mask(
        mask_size=mask_size,
        radius=radius_opposite.to('pixel').value,
        center=position[::-1],
    )
    circular_mask = np.rot90(circular_mask, k=2)
    predictor_mask = np.logical_or(predictor_mask, circular_mask)

    return predictor_mask


def get_exclusion_mask(
    mask_size: Tuple[int, int],
    position: Tuple[float, float],
    radius_excluded: Quantity,
) -> np.ndarray:
    """
    Get a mask of the pixels that we must *not* use as predictors for
    the given target pixel at `position`.

    For simplicity, the exclusion region is a disk where we exclude
    everything inside a given radius around the `position`.

    Note: This function uses the *astropy convention* for coordinates!

    Args:
        mask_size: A tuple `(x_size, y_size)` containing the size of the
            mask (in pixels) to be created.
        position: The position (in astropy = matplotlib coordinates) for
            which to compute the exclusion mask.
        radius_excluded: The radius (as an `astropy.units.Quantity` that
            can be converted to pixels) around `position` inside which
            pixels are excluded from being used as a predictor.

    Returns:
        A 2D numpy array containing the (binary) exclusion mask for the
        pixel at the given `position`.
    """

    # Create exclusion mask; flip position because get_circle_mask() uses
    # the numpy coordinate convention
    exclusion_mask = get_circle_mask(
        mask_size=mask_size,
        radius=radius_excluded.to('pixel').value,
        center=position[::-1]
    )

    return exclusion_mask


def get_predictor_pixel_selection_mask(
    mask_size: Tuple[int, int],
    position: Tuple[int, int],
    radius_position: Quantity,
    radius_opposite: Quantity,
    radius_excluded: Quantity,
) -> np.ndarray:
    """
    Get the mask that selects the predictor pixels for a given position.

    Note: This function uses the *astropy convention* for coordinates!

    Args:
        mask_size: A tuple (width, height) that specifies the size of
            the mask to be created in pixels.
        position: A tuple (x, y) specifying the position for which this
            mask is created, i.e., the mask selects the pixels that are
            used as predictors for (x, y).
        radius_position: The radius (as an astropy.units.Quantity that
            can be converted to pixels) of the circular region around
            the `position` (and the mirrored position) that is used in
            the get_predictor_mask() function.
        radius_opposite: The radius (as an astropy.units.Quantity that
            can be converted to pixels) of the circular region around
            the opposite `position`, that is, the position that we get
            if we mirror `position` across the center of the frame.
        radius_excluded: The radius (as an astropy.units.Quantity that
            can be converted to pixels) around `position` inside which
            pixels are excluded from being used as a predictor.

    Returns:
        A 2D numpy array containing a mask that selects the pixels to
        be used as predictors for the pixel at the given `position`.
    """

    # Get the mask that selects all potential predictor pixels
    predictor_mask = get_predictor_mask(
        mask_size=mask_size,
        position=position,
        radius_position=radius_position,
        radius_opposite=radius_opposite,
    )

    # Get exclusion mask (i.e., pixels we must not use as predictors)
    exclusion_mask = get_exclusion_mask(
        mask_size=mask_size,
        position=position,
        radius_excluded=radius_excluded,
    )

    # Create the actual selection mask by removing the exclusion mask
    # from the predictor mask
    selection_mask = np.logical_and(
        np.logical_not(exclusion_mask), predictor_mask
    )

    return np.asarray(selection_mask)


# -----------------------------------------------------------------------------
# OTHER MASKING-RELATED FUNCTIONS
# -----------------------------------------------------------------------------

def get_positions_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Convert a numpy mask into a list of positions selected by that mask.

    Note: The returned positions follow the numpy convention!

    Args:
        mask: A numpy array containing only boolean values (or values
            that can be interpreted as such).

    Returns:
        A sorted list of all positions (x, y) with mask[x, y] == True.
    """

    return sorted(list((x, y) for x, y in zip(*np.where(mask))))


def get_partial_roi_mask(
    roi_mask: np.ndarray, roi_split: int, n_roi_splits: int
) -> np.ndarray:
    """
    Take a `roi_mask` and return a mask that selects only a subset of
    the ROI that is specified by the number of `n_roi_splits` and the
    index `roi_split`. This function processing data in parallel, for
    example on a cluster.

    Args:
        roi_mask: A 2D numpy array containing a binary mask.
        roi_split: The index of the split for which to return the mask.
        n_roi_splits: The (total) number of splits into which the ROI
            should be divided.

    Returns:
        A 2D numpy array containing a mask that selects a subset of the
        original ROI mask, as specified above.
    """

    # Get the positions in the ROI that correspond to the current split
    positions = get_positions_from_mask(roi_mask)[roi_split::n_roi_splits]

    # Create a new mask where only those positions are True
    roi_split_mask = np.full_like(roi_mask, False)
    for (x, y) in positions:
        roi_split_mask[x, y] = True

    return roi_split_mask


def remove_connected_components(
    mask: np.ndarray,
    minimum_size: Optional[int] = None,
    maximum_size: Optional[int] = None,
) -> np.ndarray:
    """
    Remove connected components from a binary mask based on their size.

    Args:
        mask: Binary 2D numpy array from which to remove components.
        minimum_size: Components with *less* pixels than this number
            will be removed from `mask`. Set to None to not remove
            small components.
        maximum_size: Components with *more* pixels than this number
            will be removed from `mask`. Set to None to not remove
            large components.

    Returns:
        The original `mask`, with connected components removed according
        to `minimum_size` and `maximum_size`.
    """

    # Ensure that the mask is a binary
    if not np.allclose(mask, mask.astype(bool)):
        raise ValueError('Input image must be binary!')

    # Find connected components
    output, n_components = ndimage.label(mask)
    component_sizes = ndimage.sum(mask, output, range(n_components + 1))

    # Remove everything that is smaller than the minimum size
    if minimum_size is not None:
        too_small = component_sizes < minimum_size
        remove_pixel = too_small[output]
        output[remove_pixel] = 0

    # Remove everything that is larger than the maximum size
    if maximum_size is not None:
        too_large = component_sizes > maximum_size
        remove_pixel = too_large[output]
        output[remove_pixel] = 0

    return np.asarray(output).astype(bool)


def mask_frame_around_position(
    frame: np.ndarray,
    position: Tuple[float, float],
    radius: float = 5,
) -> np.ndarray:
    """
    Create a circular mask with the given `radius` at the given position
    and set the frame outside this mask to zero. This is sometimes
    required for the Gaussian2D-based photometry methods to prevent the
    Gaussian to try and fit some part of the data that is far from the
    target `position`.

    Args:
        frame: A 2D numpy array of shape `(width, height)` containing
            the data on which to run the aperture photometry.
        position: A tuple `(x, y)` specifying the position at which to
            estimate the flux. The position should be in astropy /
            photutils coordinates.
        radius: The radius of the mask; this should approximately match
            the size of a planet signal.

    Returns:
        A masked version of the given `frame` on which we can perform
        photometry based on fitting a 2D Gaussian to the data.
    """

    # Define shortcuts
    frame_size = (frame.shape[0], frame.shape[1])
    masked_frame = np.array(np.copy(frame))

    # Get circle mask; flip the position because numpy convention
    circle_mask = get_circle_mask(
        mask_size=frame_size, radius=radius, center=position[::-1]
    )

    # Apply the mask
    masked_frame[~circle_mask] = 0

    return masked_frame
