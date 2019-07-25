"""
Half-Sibling Regression model.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

from hsr4hci.utils.masking import get_annulus_mask
from typing import List, Tuple


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_roi_pixels(mask_size: Tuple[int, int],
                   pixscale: float,
                   inner_exclusion_radius: float,
                   outer_exclusion_radius: float) -> List[tuple]:
    """
    Get an iterator for the pixels (positions) in the region of interest.

    Args:
        mask_size: (Spatial) size of the input stack.
        pixscale: Pixel scale (relation between pixels and arcseconds).
        inner_exclusion_radius: Size (radius) of the inner exclusion
            region (in arcseconds).
        outer_exclusion_radius: Size (radius) of the outer exclusion
            region (in arcseconds).

    Returns:
        A list of tuples (x, y) of the positions in the ROI.
    """
    
    # Get exclusion radii of ROI and convert from arcsec to pixels
    inner_exclusion_radius = inner_exclusion_radius / pixscale
    outer_exclusion_radius = outer_exclusion_radius / pixscale
    
    # Get a binary mask for the ROI
    roi_mask = get_annulus_mask(mask_size=mask_size,
                                inner_exclusion_radius=inner_exclusion_radius,
                                outer_exclusion_radius=outer_exclusion_radius)
    
    # Convert this mask into a list of (x, y)-tuples for explicit looping
    roi_pixels = list(zip(*np.where(roi_mask)))

    return roi_pixels
