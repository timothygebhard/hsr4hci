"""
Tools for classical angular differential imaging such as frame rotation and
residual creation
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from scipy.ndimage import rotate

# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------


def de_rotate_frames(stack: np.ndarray,
                     para_angles: np.ndarray) -> np.ndarray:

    rotated = np.zeros_like(stack)
    for i in range(stack.shape[0]):
        rotated[i, :, :] = rotate(np.nan_to_num(stack[i, :, :]),
                                  -para_angles[i],
                                  reshape=False)

    return rotated


def classical_adi(stack: np.ndarray,
                  para_angles: np.ndarray,
                  mean=False) -> np.ndarray:

    # Create the PSF model
    if mean:
        psf_frame = np.mean(stack, axis=0)
    else:
        psf_frame = np.median(stack, axis=0)

    subtracted = stack - psf_frame

    residual_frames = de_rotate_frames(subtracted, para_angles)

    return np.mean(residual_frames, axis=0)




