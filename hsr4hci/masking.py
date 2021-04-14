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

from hsr4hci.coordinates import cartesian2polar, get_center
from hsr4hci.general import crop_or_pad, rotate_position, shift_image
from hsr4hci.forward_modeling import add_fake_planet


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

    Note: This function uses the *astropy convention* for coordinates!

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
    # We need to flip the `position` because get_circle_mask() uses the numpy
    # convention whereas `position` is assumed to be in the astropy convention
    circular_mask = get_circle_mask(
        mask_size=mask_size,
        radius=radius_position.to('pixel').value,
        center=position[::-1],
    )
    mask = np.logical_or(mask, circular_mask)

    # Add circular selection mask at mirror position (-x, -y)
    # We also need to flip the coordinates here to get to the numpy convention
    mirror_position = (
        2 * center[1] - position[1],
        2 * center[0] - position[0],
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
    psf_template: np.ndarray,
    signal_time: Optional[int],
    threshold: float = 2e-1,
) -> np.ndarray:
    """
    Get a mask of the pixels that we must *not* use as predictors.

    The idea of this function is the following: Instead of manually
    constructing an exclusion region based on our knowledge of the
    signal size and the planet movement (which is rather tedious),
    we simply construct a signal stack (with a low temporal resolution,
    to reduce the computational costs) for the hypothesis given by
    the tuple (position, signal_time) and the exclude those pixels
    that "know too much" about the target time series at `position`.
    For this, we compute the elementwise product between the target
    time series and all other time series and place a threshold on
    the maximum. This then excludes time series with a bump that
    overlaps too much with the bump in the target time series, which
    is what we need for the HSR to work.

    When no `signal_time` is given (for the "default" models, which work
    under the assumption that no planet is present), we simply place the
    PSF template at the given position and threshold it to determine the
    exclusion region.

    Note: This function uses the *astropy convention* for coordinates!

    Args:
        mask_size: A tuple `(x_size, y_size)` containing the size of the
            mask (in pixels) to be created.
        position: The position (in astropy = matplotlib coordinates) for
            which to compute the exclusion mask. The exclusion mask will
            mark the pixels that we must not use as predictors for the
            pixel at `position`.
        parang: A 1D numpy array containing the parallactic angles.
        psf_template: A 2D numpy array containing the (unsaturated) PSF
            template.
        signal_time: An integer that specifies the index of the frame
            in which the planet signal peaks at the given `position`.
            If signal_time is None, the function assumes that there is
            no planet signal in `position` at any time.
        threshold: Threshold value used for creating the mask. This
            parameter can be used to control how "conservative" the
            exclusion mask will be: the lower the threshold value, the
            more pixels will be excluded as predictors.

    Returns:
        A 2D numpy array containing the (binary) exclusion mask for the
        pixel at the given `position`.
    """

    # -------------------------------------------------------------------------
    # Sanity checks and preliminaries
    # -------------------------------------------------------------------------

    # Make sure that the options do not contradict each other
    if (signal_time is not None) and (signal_time < 0):
        raise ValueError('Negative signal times are not allowed!')

    # Defines shortcuts
    n_frames = len(parang)
    center = get_center(mask_size)

    # Prepare the unsaturated PSF template (crop it, normalize it)
    psf_resized = crop_or_pad(psf_template, mask_size)
    psf_resized -= np.min(psf_resized)
    psf_resized /= np.max(psf_resized)

    # -------------------------------------------------------------------------
    # CASE 1: Exclusion mask *without* signal time
    # -------------------------------------------------------------------------

    # In this scenario, we compute the exclusion mask for a pixels without
    # considering the field rotation: we only exclude pixels that are too
    # close to the given `position`. This is the mask that is used for the
    # "default" models (i.e., models that assume that no planet is present).
    if signal_time is None:

        # Shift the PSF template so that it is centered on the given position
        exclusion_mask = shift_image(
            image=psf_resized,
            offset=(
                float(position[0] - center[0]),
                float(position[1] - center[1]),
            ),
            interpolation='bilinear',
            mode='constant',
        )

        # Threshold the shifted PSF to get the exclusion mask
        exclusion_mask = exclusion_mask > threshold / 2

    # -------------------------------------------------------------------------
    # CASE 2: Exclusion mask *with* signal time
    # -------------------------------------------------------------------------

    # In case we use the field rotation (for signal masking / fitting models),
    # we (for now) simply exclude an arc with an opening angle that matches
    # the field rotation.

    else:

        # Down-sample the parallactic angle: computing the signal stack can be
        # expensive with a large number of frames, but the exclusion mask that
        # we find using this method is not very sensitive to the temporal
        # resolution anyway, so we can compute it using a quick approximation
        # of the signal stack to speed things up.
        n = n_frames // 100
        parang_resampled = parang[::n]

        # Compute final planet position under the hypothesis given by the
        # tuple (position, signal_time)
        final_position = rotate_position(
            position=position,
            center=get_center(mask_size),
            angle=float(parang[int(signal_time)]),
        )

        # Compute full signal stack under our hypothesis and normalize it to 1
        signal_stack = add_fake_planet(
            stack=np.zeros(
                (len(parang_resampled), mask_size[0], mask_size[1])
            ),
            parang=parang_resampled,
            psf_template=psf_template,
            polar_position=cartesian2polar(
                position=(final_position[0], final_position[1]),
                frame_size=mask_size,
            ),
            magnitude=1,
            extra_scaling=1,
            dit_stack=1,
            dit_psf_template=1,
            return_planet_positions=False,
        )
        signal_stack = np.asarray(signal_stack / np.max(signal_stack))

        # Get the time series for the position
        target = signal_stack[:, int(position[1]), int(position[0])]

        # Compute the "overlap" of the target time series with every other
        # time series. For this, we take the (element-wise) product of each
        # pair of time series, find the maximum, and take the square root.
        # The last step (sqrt) is so that the overlap_map has the same scale
        # as the normal unsaturated PSF template.
        overlap_map = np.einsum('i,ijk->ijk', target, signal_stack)
        overlap_map = np.max(overlap_map, axis=0)
        overlap_map = np.sqrt(overlap_map)

        # Threshold the overlap_map to get the exclusion map: all pixels whose
        # time series "know too much" about the target time series are excluded
        exclusion_mask = overlap_map > threshold

    # -------------------------------------------------------------------------
    # Apply a morphological filter to the exclusion mask and return it
    # -------------------------------------------------------------------------

    # Dilate the mask by one pixel for a little extra "safety margin"
    selem = disk(radius=1)
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
    )

    # Create the actual selection mask by removing the exclusion mask
    # from the predictor mask
    selection_mask = np.logical_and(
        np.logical_not(exclusion_mask), predictor_mask
    )

    return np.asarray(selection_mask)


def get_radial_masks(
    mask_size: Tuple[int, int],
    n_rings: int,
) -> List[np.ndarray]:
    """
    Create a list of annulus-shaped masks that can be used, for example,
    to compute radial averages.

    Source: https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_radial_mean.html

    Args:
        mask_size: A tuple `(x_size, y_size)` containing the size of the
            mask (in pixels) to be created.
        n_rings: Number of rings to be created for the mask.

    Returns:
        A list of annulus masks.
    """

    sx, sy = mask_size
    x, y = np.ogrid[0:sx, 0:sy]
    r = np.hypot(x - sx / 2, y - sy / 2)
    radial_bins = (n_rings * r / r.max()).astype(np.int)

    masks = []
    for i in range(n_rings):
        mask = radial_bins == i
        masks.append(mask)

    return masks


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
