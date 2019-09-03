"""
Select pixels in the region of interest (ROI).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

from hsr4hci.utils.masking import get_annulus_mask
from typing import Tuple


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
        mask_size: (Spatial) size of the input stack.
        pixscale: Pixel scale (relation between pixels and arcseconds).
        inner_exclusion_radius: Size (radius) of the inner exclusion
            region (in arcseconds).
        outer_exclusion_radius: Size (radius) of the outer exclusion
            region (in arcseconds).

    Returns:
        A numpy array with the mask of the pixels in the ROI.
    """
    
    # Get exclusion radii of ROI and convert from arcsec to pixels
    inner_exclusion_radius = inner_exclusion_radius / pixscale
    outer_exclusion_radius = outer_exclusion_radius / pixscale
    
    # Get a binary mask for the ROI
    roi_mask = get_annulus_mask(mask_size=mask_size,
                                inner_exclusion_radius=inner_exclusion_radius,
                                outer_exclusion_radius=outer_exclusion_radius)

    return roi_mask
