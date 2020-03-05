"""
Utility functions for the E/M-style iterative version of HSR.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

from hsr4hci.utils.derotating import derotate_frames


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    """
    Convenience wrapper for a simple ReLU function (cutoff values < 0).
    """
    return np.maximum(x, 0)


def get_signal_estimate_stack(signal_estimate: np.ndarray,
                              parang: np.ndarray) -> np.ndarray:
    """
    Construct a signal stack from the signal_estimate and rotate it into
    the coordinate system of the original stack from which we are going
    to subtract it during the EM-style HSR.

    Args:
        signal_estimate: A 2D numpy array of size (width, height)
            containing the current  estimate for the planet signal.
        parang: A 1D numpy array of size (n_frames, ) containing the
            parallactic angles of the original stack.

    Returns:
        A 3D numpy array of shape (n_frames, width, height) where each
        frames contains the `signal_estimate` rotated by the respective
        parallactic angle.
    """

    # Create as many copies of the signal_estimate as their are frames
    signal_estimate_stack = [signal_estimate for _ in range(len(parang))]
    signal_estimate_stack = np.array(signal_estimate_stack)

    # Rotate each signal frame to its corresponding parallactic angle
    signal_estimate_stack = derotate_frames(stack=signal_estimate_stack,
                                            parang=(-1.0 * parang))

    return signal_estimate_stack
