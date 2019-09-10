"""
Select pixels in the region of interest (ROI).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Tuple

import numpy as np

from hsr4hci.utils.masking import get_annulus_mask


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_roi_mask(mask_size: Tuple[int, int],
                 pixscale: float,
                 inner_exclusion_radius: float,
                 outer_exclusion_radius: float) -> np.ndarray:
    """
    Get a numpy array masking the pixels within the region of interest.

    Args:
        mask_size: A tuple (width, height) containing the spatial size
            of the input stack.
        pixscale: The (instrument-specific) pixel scale which defines
            the relation between pixels and arcseconds.
        inner_exclusion_radius: Radius of the inner exclusion region
            (in arcseconds).
        outer_exclusion_radius: Radius of the outer exclusion region
            (in arcseconds).

    Returns:
        A 2D numpy array of size `mask_size` which masks the pixels
        within the specified region of interest.
    """

    # Get exclusion radii of ROI and convert from arcseconds to pixels
    inner_exclusion_radius = inner_exclusion_radius / pixscale
    outer_exclusion_radius = outer_exclusion_radius / pixscale

    # Get a binary mask for the ROI
    roi_mask = get_annulus_mask(mask_size=mask_size,
                                inner_radius=inner_exclusion_radius,
                                outer_radius=outer_exclusion_radius)

    return roi_mask
