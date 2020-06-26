"""
Utility functions for (de)-rotating stacks and computing residuals.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Optional

from joblib import delayed, Parallel
from scipy.ndimage import rotate

import bottleneck as bn
import numpy as np


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def derotate_frames(
    stack: np.ndarray,
    parang: np.ndarray,
    mask: Optional[np.ndarray] = None,
    order: int = 3,
    n_processes: int = 4,
) -> np.ndarray:
    """
    Derotate all frames in the stack by their parallactic angle.

    Args:
        stack: Stack of frames to be de-rotated.
        parang: Array of parallactic angles (one for each frame).
        mask: Mask to apply after derotating. Usually, pixels for which
            there exist no real values are set to NaN. However, for
            derotating, these have to be casted to zeros (otherwise the
            interpolation turns everything into a NaN). This mask here
            allows to restore these NaN values again. Note that this
            mask selects the pixels that will be set to NaN; that means,
            for example, the usual ROI mask should be inverted before it
            is passed to this function.
        order: The order of the spline interpolation for the rotation.
            Has to be in the range [0, 5]; default is 3.
        n_processes: Number of parallel processes to be used to derotate
            the frames in parallel; default is 4.

    Returns:
        The stack with every frame derotated by its parallactic angle.
    """

    # Define a helper function that defines the rotation for a single frame;
    # this is only a partial function application of scipy.ndimage.rotate()
    def rotate_frame(frame: np.ndarray, angle: float) -> np.ndarray:
        return rotate(input=frame, angle=angle, reshape=False, order=order)

    # Either derotate frames in parallel using joblib...
    if n_processes > 1:
        with Parallel(n_jobs=n_processes, require='sharedmem') as run:
            derotated = run(
                delayed(rotate_frame)(frame, angle)
                for frame, angle in zip(np.nan_to_num(stack), -parang)
            )

    # ...or simply process frames serially
    else:
        derotated = [
            rotate_frame(frame, angle)
            for frame, angle in zip(np.nan_to_num(stack), -parang)
        ]

    # Convert result to numpy array
    derotated = np.array(derotated)

    # Check if there is a mask that we need to apply after derotating
    if mask is not None:
        derotated[:, mask] = np.nan

    return derotated


def derotate_combine(
    stack: np.ndarray,
    parang: np.ndarray,
    mask: Optional[np.ndarray] = None,
    order: int = 3,
    subtract: Optional[str] = None,
    combine: str = 'mean',
    n_processes: int = 4,
) -> np.ndarray:
    """
    Derotate all frames in the stack and combine (= average) them.

    Args:
        stack: A 3D numpy array of shape (n_frames, width, height)
            containing the stack of (residual) frames.
        parang: A 1D numpy array of shape (n_frames, ) containing the
            respective parallactic angle for each frame.
        mask: Mask to apply after derotating. Usually, pixels for which
            there exist no real values are set to NaN. However, for
            derotating, these have to be casted to zeros (otherwise the
            interpolation turns everything into a NaN). This mask here
            allows to restore these NaN values again.
        order: The order of the spline interpolation for the rotation.
            Has to be in the range [0, 5]; default is 3.
        subtract: A string specifying what to subtract from the stack
            before derotating the frames. Options are "mean", "median"
            or None.
        combine: A string specifying how to combine the frames after
            derotating them. Options are "mean" or "median".
        n_processes: Number of parallel processes to be used to derotate
            the frames in parallel; default is 4.

    Returns:
        A 2D numpy array of shape (width, height) containing the
        derotated and combined stack.
    """

    # Create the classical ADI estimate of the PSF (either as the mean or
    # median along the time axis of the frames before de-rotating)
    if subtract == 'mean':
        psf_frame = np.nanmean(stack, axis=0)
    elif subtract == 'median':
        psf_frame = np.nanmedian(stack, axis=0)
    elif subtract is None:
        psf_frame = np.zeros(stack.shape[1:])
    else:
        raise ValueError('Illegal option for parameter "subtract"!')

    # Subtract the PSF estimate from every frame
    subtracted = stack - psf_frame

    # De-rotate all frames by their respective parallactic angles
    residual_frames = derotate_frames(
        stack=subtracted, parang=parang, order=order, n_processes=n_processes,
    )

    # Combine derotated frames either by taking the mean or median
    if combine == 'mean':
        result = bn.nanmean(residual_frames, axis=0)
    elif combine == 'median':
        result = bn.nanmedian(residual_frames, axis=0)
    else:
        raise ValueError('Illegal option for parameter "combine"!')

    # Apply mask to result before returning it
    if mask is not None:
        result[mask] = np.nan

    return result
