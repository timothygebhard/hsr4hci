"""
Tools for classical angular differential imaging (ADI), such as frame
(de)-rotation and residual creation.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

from scipy.ndimage import rotate
from tqdm import tqdm

from typing import Optional


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def derotate_frames(stack: np.ndarray,
                    parang: np.ndarray,
                    mask: Optional[np.ndarray] = None,
                    verbose: bool = False) -> np.ndarray:
    """
    Derotate all frames in the stack by their parallactic angle.

    Args:
        stack: Stack of frames to be de-rotated.
        parang: Array of parallactic angles (one for each frame).
        mask: Mask to apply after derotating. Usually, pixels for which
            there exists no prediction are set to NaN. However, for
            derotating, these have to be casted to zeros (otherwise the
            interpolation turns everything into a NaN). This mask here
            allows to restore these NaN values again.
        verbose: Whether or not to print a progress bar.

    Returns:
        The stack with every frame derotated by its parallactic angle.
    """

    # Initialize array that will hold the de-rotated frames
    derotated = np.zeros_like(stack)

    # If desired, use a tqdm decorator to show the progress
    if verbose:
        indices = tqdm(iterable=range(stack.shape[0]), ncols=80)
    else:
        indices = range(stack.shape[0])

    # Loop over all frames and derotate them by their parallactic angle
    for i in indices:
        derotated[i, :, :] = rotate(input=np.nan_to_num(stack[i, :, :]),
                                    angle=-parang[i],
                                    reshape=False)

    # Check if there is a mask that we need to apply after derotating
    if mask is not None and mask.shape[1:] == stack.shape[1:]:
        derotated[:, mask] = np.nan

    return derotated


def classical_adi(stack: np.ndarray,
                  para_angles: np.ndarray,
                  mean: bool = False) -> np.ndarray:
    """
    Perform classical ADI on the given input stack.
    
    Args:
        stack: ADI stack of frames.
        para_angles: Array of parallactic angles.
        mean: If True, use the mean along the time axis as the estimate
            for the PSF which is subtracted from all frames. Otherwise,
            use the median.

    Returns:
        The classical ADI post-processing result of the input stack.
    """

    # Create the PSF estimate (either as the mean or median along the
    # time axis of the frames before de-rotating)
    if mean:
        psf_frame = np.mean(stack, axis=0)
    else:
        psf_frame = np.median(stack, axis=0)

    # Subtract the PSF estimate from every frame
    subtracted = stack - psf_frame

    # De-rotate all frames by their respective parallactic angles
    residual_frames = derotate_frames(subtracted, para_angles)

    # TODO: Should we be able to choose between mean and median here?
    # Combine the residual frames by averaging along the time axis
    return np.mean(residual_frames, axis=0)
