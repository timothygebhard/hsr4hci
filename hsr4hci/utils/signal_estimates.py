"""
Utility functions for creating signal estimates.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import List, Tuple
from warnings import warn

from skimage.filters import threshold_minimum
from skimage.morphology import disk, opening
from tqdm.auto import tqdm

import numpy as np

from hsr4hci.utils.derotating import derotate_combine


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_signal_estimate(
    parang: np.ndarray,
    match_fraction: np.ndarray,
    default_residuals: np.ndarray,
    non_default_residuals: np.ndarray,
    roi_mask: np.ndarray,
    filter_size: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Threshold the given `match_fraction` and use the resulting mask to
    construct a signal estimate from the given residuals.

    Args:
        parang:
        match_fraction:
        default_residuals:
        non_default_residuals:
        roi_mask:
        filter_size:

    Returns:
        A 3-tuple consisting of the `signal_estimate`, the mask that was
        used to create it, and the threshold that was used to obtain the
        mask.
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

    # Use the filtered mask to decide for which pixels the "default" model
    # is used, and for which the planet-based model is used.
    residuals = np.copy(default_residuals)
    residuals[:, filtered_mask] = np.array(
        non_default_residuals[:, filtered_mask]
    )

    # Compute signal estimate and store it
    signal_estimate = derotate_combine(residuals, parang, mask=~roi_mask)

    return signal_estimate, filtered_mask, threshold


def get_signal_estimates_and_masks(
    parang: np.ndarray,
    match_fraction: np.ndarray,
    default_residuals: np.ndarray,
    signal_masking_residuals: np.ndarray,
    roi_mask: np.ndarray,
    filter_size: int = 0,
    n_thresholds: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function wraps the loop over different threshold values, the
    creation (and morphological filtering) of selection masks, the
    assembly of residuals (by combining "default" and signal masking-
    based residuals based on the masks), and finally the computation
    of signal estimates.

    Args:
        parang: A 1D numpy array of shape `(n_frames, )` containing the
            parallactic angles.
        match_fraction: A 2D numpy array of shape `(width, height)` that
            contains the match fraction for each pixel -- this is the
            quantity that will be thresholded to compute the selection
            masks for choosing which residual to use for which pixel.
        default_residuals: A 3D, stack-like numpy array that contains
            the "default" residuals.
        signal_masking_residuals: A 3D, stack-like numpy array that
            contains signal masking-based residuals.
        roi_mask: A 2D numpy array of shape `(width, height)` containing
            the ROI mask (for masking the signal estimates).
        filter_size: Size parameter for the morphological opening filter
            that can be used to remove individual pixels in the masks
            that are obtained by thresholding the match fraction.
        n_thresholds: Number of threshold values for which to run.

    Returns:
        A tuple of numpy arrays, consisting of:
        signal_estimates, thresholds, thresholded_masks, filtered_masks
    """

    # Define threshold values
    thresholds = np.linspace(0, 1, n_thresholds + 1)
    thresholds = np.insert(thresholds, -1, [-1, 0, 1])
    thresholds = np.array(sorted(np.unique(thresholds)))

    # Keep track of the masks and signal estimates that we generate
    thresholded_masks: List[np.ndarray] = []
    filtered_masks: List[np.ndarray] = []
    signal_estimates: List[np.ndarray] = []

    for threshold in tqdm(thresholds, ncols=80):

        # Threshold the matching fraction
        thresholded_mask = match_fraction > threshold
        thresholded_masks.append(thresholded_mask)

        # Define a structure element and apply a morphological filter (more
        # precisely, an opening filter) to remove small regions in the mask.
        # This reflects our knowledge that a true planet path should have the
        # characteristic "sausage"-shape, and not consist of single pixels
        structure_element = disk(filter_size)
        filtered_mask = np.logical_and(
            opening(thresholded_mask, structure_element), thresholded_mask
        )
        filtered_masks.append(filtered_mask)

        # Select the residuals: by default, use default model for everything.
        # Only for the pixels selected by the filtered mask do we choose the
        # residuals from the best model based on signal masking.
        residuals = np.copy(default_residuals)
        residuals[:, filtered_mask] = np.array(
            signal_masking_residuals[:, filtered_mask]
        )

        # Compute signal estimate and store it
        signal_estimate = derotate_combine(residuals, parang)
        signal_estimate[~roi_mask] = np.nan
        signal_estimates.append(signal_estimate)

    return (
        np.array(signal_estimates),
        np.array(thresholds),
        np.array(thresholded_masks),
        np.array(filtered_masks),
    )
