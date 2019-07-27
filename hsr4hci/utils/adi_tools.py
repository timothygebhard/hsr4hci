"""
Tools for classical angular differential imaging (ADI), such as frame
(de)-rotation and residual creation.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from scipy.ndimage import rotate


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def derotate_frames(stack: np.ndarray,
                    para_angles: np.ndarray) -> np.ndarray:
    """
    De-rotate every frame in the given stack by its respective
    parallactic angle.
    
    Args:
        stack: Stack of frames to be de-rotated.
        para_angles: Array of parallactic angles (one for each frame).

    Returns:
        The stack with every frame de-rotated by its parallactic angle.
    """

    # Initialize array that will hold the de-rotated frames
    derotated = np.zeros_like(stack)

    # Loop over all frames and de-rotate them by their parallactic angle
    for i in range(stack.shape[0]):
        derotated[i, :, :] = rotate(input=np.nan_to_num(stack[i, :, :]),
                                    angle=-para_angles[i],
                                    reshape=False)

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
