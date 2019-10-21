"""
Utility functions for selecting predictors.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from cmath import polar
from typing import Optional

import numpy as np

from hsr4hci.utils.masking import get_annulus_mask, get_circle_mask


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_default_mask(mask_size: tuple,
                     position: tuple,
                     n_regions: int = 1,
                     region_size: Optional[int] = None,
                     **_) -> np.ndarray:
    """
    Get a mask to select the predictor pixels for the given `position`.

    This function implements a particular choice for the predictors
    where `n_regions + 1` circular apertures of size `region_size` are
    placed uniformly on an imaginary circle around the center of the
    frame with radius such that it passes through the given `position`.
    The apertures are aligned such that one of them is centered on the
    `position`. The predictor mask will then contain all apertures but
    the one on `position`.
    Example: For `n_regions=1` (the default), we only select a circular
    region of radius `region_size` that is centered on the pixel that
    is obtained by mirroring the `position` across the frame center.
    This choice is motivated by the fact that we know from theory that
    speckles are always point-symmetric across the origin (due to the
    way that the adaptive optics system works), meaning that this
    region should be useful as a predictor for the systematics that we
    would like to remove from the frames.

    Args:
        mask_size: A tuple (width, height) containing the spatial size
            of the input stack.
        position: A tuple (x, y) containing the position for which to
            create the predictor mask.
        n_regions: Number of re
        region_size:

    Returns:

    """

    # Initialize an empty mask and compute its center
    mask = np.zeros(mask_size).astype(np.bool)
    center = (mask_size[0] / 2, mask_size[1] / 2)

    # Compute polar representation of the position
    radius, phi = polar(complex(position[1] - center[1],
                                position[0] - center[0]))

    # Compute the angle between two regions
    theta = np.deg2rad(360 / (n_regions + 1))

    # Loop over regions to be created
    for i in range(n_regions):

        new_complex_position = radius * np.exp(1j * (phi + (i + 1) * theta))
        new_position = (int(np.imag(new_complex_position) + center[0]),
                        int(np.real(new_complex_position) + center[1]))

        if region_size is None:
            mask[new_position] = True
        else:
            disk = get_circle_mask(mask_size=mask_size,
                                   radius=region_size,
                                   center=new_position)
            mask = np.logical_or(mask, disk)

    return mask


def get_default_grid_mask(mask_size: tuple,
                          position: tuple,
                          lambda_over_d: float,
                          pixscale: float,
                          n_regions: int = 1,
                          region_size: Optional[int] = None,
                          exclusion_radius: float = 5,
                          **_) -> np.ndarray:

    # Compute lambda / D in units of pixels
    lod_pixels = lambda_over_d / pixscale

    # Get the normal default mask
    mask = get_default_mask(mask_size=mask_size,
                            position=position,
                            n_regions=n_regions,
                            region_size=region_size)

    # Additionally, add a grid to the mask
    xx, yy = np.meshgrid(np.arange(0, mask_size[0], 2 * int(lod_pixels))[1:-1],
                         np.arange(0, mask_size[1], 2 * int(lod_pixels))[1:-1])
    mask[xx, yy] = True

    # Exclude everything in a given region around the position
    exclusion_mask = get_circle_mask(mask_size=mask_size,
                                     radius=(exclusion_radius * lod_pixels),
                                     center=position)
    mask[exclusion_mask] = False
    
    return mask


def get_santa_mask(mask_size: tuple,
                   position: tuple,
                   lambda_over_d: float,
                   pixscale: float,
                   **_) -> np.ndarray:

    # Compute lambda / D in units of pixels
    lod_pixels = lambda_over_d / pixscale

    # Compute mask center and separation of position from the center
    center = tuple([_ / 2 for _ in mask_size])
    separation = np.hypot((position[0] - center[0]), (position[1] - center[1]))

    # Initialize an empty mask of the desired size
    mask = np.full(mask_size, False)

    # Add circular selection mask at position (radius: 2 lambda over D)
    circular_mask = get_circle_mask(mask_size=mask_size,
                                    radius=(2 * lod_pixels),
                                    center=position)
    mask = np.logical_or(mask, circular_mask)

    # Add circular selection  mask at mirror position (radius: 2 lambda over D)
    mirror_position = tuple([2 * center[i] - position[i] for i in range(2)])
    circular_mask = get_circle_mask(mask_size=mask_size,
                                    radius=(2 * lod_pixels),
                                    center=mirror_position)
    mask = np.logical_or(mask, circular_mask)

    # Add annulus-shaped selection mask (width: 1 lambda over D)
    annulus = get_annulus_mask(mask_size=mask_size,
                               inner_radius=(separation - 0.5 * lod_pixels),
                               outer_radius=(separation + 0.5 * lod_pixels))
    mask = np.logical_or(mask, annulus)

    return mask
