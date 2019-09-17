"""
Utility functions for selecting predictors (and targets).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from cmath import polar
from typing import Optional

import cv2
import numpy as np

from hsr4hci.utils.masking import get_annulus_mask, get_circle_mask


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_predictor_mask(mask_type: str,
                       mask_args: dict) -> np.ndarray:

    # Collect keyword arguments for default mask
    kwargs = {**dict(mask_size=mask_args['mask_size'],
                     position=mask_args['position']),
              **mask_args['mask_params']}

    # If we are using the default mask, we are done here
    if mask_type == 'default':
        return get_default_mask(**kwargs)

    # For the other mask types, we need to add some more options
    kwargs['lambda_over_d'] = mask_args['lambda_over_d']
    kwargs['pixscale'] = mask_args['pixscale']

    # Return either the default_grid mask or the santa mask
    if mask_type == 'default_grid':
        return get_default_grid_mask(**kwargs)
    if mask_type == 'santa':
        return get_santa_mask(**kwargs)

    # For unknown mask types, raise an error
    raise ValueError('Invalid choice for mask_type!')


def get_default_mask(mask_size: tuple,
                     position: tuple,
                     n_regions: int = 1,
                     region_size: Optional[int] = None) -> np.ndarray:
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
                          exclusion_radius: float = 5) -> np.ndarray:

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
                   pixscale: float) -> np.ndarray:

    # Compute lambda / D in units of pixels
    lod_pixels = lambda_over_d / pixscale

    mask = np.zeros(mask_size).astype(np.bool)
    center = (mask_size[0] / 2, mask_size[1] / 2)

    # Compute polar representation of the position
    radius, phi = polar(complex(position[1] - center[1],
                                position[0] - center[0]))

    # Define default options of ellipses
    ellipse_options = dict(angle=np.rad2deg(phi) + 90,
                           startAngle=0,
                           endAngle=360,
                           color=(1, 1, 1),
                           thickness=-1)

    # Add mask at position
    ellipse = np.zeros(mask_size)
    cv2.ellipse(img=ellipse,
                center=tuple(map(int, position[::-1])),
                axes=tuple(map(int, (5 * lod_pixels, 4 * lod_pixels))),
                **ellipse_options)
    ellipse = ellipse.astype(bool)
    mask[ellipse] = True

    # Add disk mask at mirror position
    new_complex_position = radius * np.exp(1j * (phi + np.pi))
    new_position = (int(np.imag(new_complex_position) + center[0]),
                    int(np.real(new_complex_position) + center[1]))
    ellipse = np.zeros(mask_size)
    cv2.ellipse(img=ellipse,
                center=tuple(map(int, new_position[::-1])),
                axes=tuple(map(int, (4.5 * lod_pixels, 3 * lod_pixels))),
                **ellipse_options)
    ellipse = ellipse.astype(bool)
    mask[ellipse] = True

    # Add annulus around frame center
    annulus = get_annulus_mask(mask_size=mask_size,
                               inner_radius=(radius - lod_pixels),
                               outer_radius=(radius + lod_pixels))
    mask = np.logical_or(mask, annulus)

    # Define exclusion region around the position
    exclusion_mask = np.zeros(mask_size)
    cv2.ellipse(img=exclusion_mask,
                center=tuple(map(int, position[::-1])),
                axes=tuple(map(int, (5 * lod_pixels, 2.5 * lod_pixels))),
                **ellipse_options)
    exclusion_mask = exclusion_mask.astype(bool)
    mask[exclusion_mask] = False

    return mask
