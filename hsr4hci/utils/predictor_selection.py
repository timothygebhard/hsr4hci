"""
Utility functions for selecting predictors (and targets).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Optional

import numpy as np

from hsr4hci.utils.masking import get_circle_mask


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_predictor_mask(mask_size: tuple,
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

    mask = np.zeros(mask_size)
    center = (mask_size[0] / 2, mask_size[1] / 2)

    complex_position = ((position[0] - center[0]) * 1j +
                        (position[1] - center[1]))
    r, phi = np.abs(complex_position), np.angle(complex_position)

    theta = np.deg2rad(360 / (n_regions + 1))

    # Loop over regions to be created
    for i in range(n_regions):

        new_complex_position = r * np.exp(1j * (phi + (i + 1) * theta))
        new_position = (int(np.imag(new_complex_position) + center[0]),
                        int(np.real(new_complex_position) + center[1]))

        if region_size is None:
            mask[new_position] = 1
        else:
            disk = get_circle_mask(mask_size=mask_size,
                                   radius=region_size,
                                   center=new_position)
            mask = np.logical_or(mask, disk)

    return mask
