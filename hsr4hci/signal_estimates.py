"""
Utility functions for creating signal estimates.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Tuple
from warnings import warn

from skimage.filters import threshold_minimum
from skimage.morphology import disk, opening

import numpy as np


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_selection_mask(
    match_fraction: np.ndarray,
    roi_mask: np.ndarray,
    filter_size: int = 0,
) -> Tuple[np.ndarray, float]:
    """
    Threshold the match fraction and apply a morphological filter to
    create the selection mask for choosing residuals.

    Args:
        match_fraction:
        roi_mask:
        filter_size:

    Returns:
        A tuple (`selection_mask` `threshold`).
    """

    # Determine the "optimal" threshold for the match fraction
    threshold = threshold_minimum(np.nan_to_num(match_fraction[roi_mask]))

    # Apply threshold to match fraction to get a mask
    mask = match_fraction >= threshold

    # If the mask selects "too many" pixels (i.e., more than can reasonably
    # be affected by planet signals), fall back to the default. This is a
    # somewhat crude way to incorporate our knowledge that a real planet
    # signal must be spatially sparse.
    if np.mean(mask[roi_mask]) > 0.2:
        mask = np.full(mask.shape, False)
        warn('Threshold allows too many pixels, falling back to default mask!')

    # Define a structure element and apply a morphological filter (more
    # precisely, an opening filter) to remove small regions in the mask.
    structure_element = disk(filter_size)
    filtered_mask = np.logical_and(opening(mask, structure_element), mask)

    return filtered_mask, threshold
