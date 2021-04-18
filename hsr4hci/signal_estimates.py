"""
Utility functions for creating signal estimates.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Tuple

from scipy import ndimage
from skimage.filters import threshold_isodata

import numpy as np

from hsr4hci.masking import get_circle_mask


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_selection_mask(
    match_fraction: np.ndarray,
    roi_mask: np.ndarray,
    filter_size: int = 0,
    minimum_size: int = 25,
) -> Tuple[np.ndarray, float]:
    """
    Threshold the match fraction and apply a filter to create the
    selection mask for choosing residuals.

    FIXME: Ultimately, we probably want to use some form of segmentation
           or local thresholding here; not one single global threshold.

    Args:
        match_fraction:
        roi_mask:
        filter_size:
        minimum_size:

    Returns:
        A tuple (`selection_mask`, `threshold`).
    """

    mask_size = (int(roi_mask.shape[0]), int(roi_mask.shape[1]))

    # Drop the innermost few pixels, where we realistically cannot find
    # planets but which for signal fitting are often very bright, which breaks
    # the threshold estimation
    drop_mask = get_circle_mask(mask_size=mask_size, radius=4)

    # For threshold_isodata(), it seems to make sense to remove all the pixels
    # that are (close to) zero in the match fraction (?)
    zero_mask = np.isclose(match_fraction, 0)

    # Select pixels on which to compute the threshold
    pixel_mask = np.logical_and(roi_mask, ~np.logical_or(drop_mask, zero_mask))
    pixels = np.nan_to_num(match_fraction[pixel_mask])

    # Determine the "optimal" threshold for the match fraction
    try:
        threshold = threshold_isodata(pixels)
    except RuntimeError:
        threshold = 1

    # Apply threshold to match fraction to get a mask
    mask = match_fraction >= threshold

    # Drop regions in the mask that are below a certain size
    final_mask, nb_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, final_mask, range(nb_labels + 1))
    mask_size = sizes < minimum_size
    remove_pixel = mask_size[final_mask]
    final_mask[remove_pixel] = 0
    final_mask = final_mask > 0

    return final_mask, threshold
