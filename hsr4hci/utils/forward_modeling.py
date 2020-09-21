"""
Utility functions for forward modeling (necessary for toy data sets!).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from cmath import polar
from typing import Tuple

import numpy as np

from hsr4hci.utils.general import add_array_with_interpolation


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_signal_stack(
    position: Tuple[float, float],
    frame_size: Tuple[int, int],
    parang: np.ndarray,
    psf_cropped: np.ndarray,
) -> Tuple[np.ndarray, list]:
    """
    Compute the forward model: Assume that at time t=0, the planet is
    at the given `position. The apparent motion of the sky (and thus `
    the planet) is given by the parallactic angles in `parang`. This
    function now takes the provided cropped and masked PSF. Then, an
    empty stack of size `(len(parang), *frame_size)` is created, and
    for each time t, the cropped PSF template is added into the stack
    at the correct position according to parang[t].

    Args:
        position: A tuple containing the initial planet position (i.e.,
            we compute the forward model under the assumption that the
            planet is at this position for t=0).
        frame_size: A tuple with the spatial size `(x_size, y_size)`
            of the forward model stack to be created.
        parang: A numpy array containing the parallactic angle for
            every frame.
        psf_cropped: A numpy array containing the cropped and masked
            PSF template, as it is returned by `crop_psf_template()`.

    Returns:
        A tuple consisting of a numpy array containing the forward model
        (i.e., a stack of frames where every frame contains only the
        "planet signal" at the correct position), computed under the
        given assumptions, as well as a list of tuples (x, y) containing
        the position of the planet in the forward model for every frame.
    """

    # Define some shortcuts
    n_frames = len(parang)
    frame_width, frame_height = frame_size
    frame_center = (frame_width / 2, frame_height / 2)

    # Compute relative rotation angles (i.e., the rotation relative to t=0)
    rotation_angles = parang - parang[0]

    # Compute polar representation of initial position
    r, phi = polar(complex(position[1] - frame_center[1],
                           position[0] - frame_center[0]))

    # Initialize empty signal stack and list of planet positions
    signal_stack = np.zeros((n_frames, frame_width, frame_height))
    planet_positions = list()

    # Loop over all frames and add the cropped and masked PSF template at
    # the position that corresponds to the respective parallactic angle
    for i in range(n_frames):

        # Compute injection position for the current frame: Rotate the initial
        # position by the correct angle (using the exponential representation
        # of 2D vectors) and convert the resulting position back to Cartesian
        # coordinates. Store this value to a list that we will return along
        # with the signal stack we have constructed.
        theta = np.deg2rad(rotation_angles[i])
        new_complex_position = r * np.exp(1j * (phi - theta))
        injection_position = (np.imag(new_complex_position) + frame_center[0],
                              np.real(new_complex_position) + frame_center[1])
        planet_positions.append(injection_position)

        # Add cropped and masked PSF template at the injection_position.
        # The add_array_with_interpolation() function is able to automatically
        # deal with cases where the injection_position is a tuple of floats,
        # or the PSF template exceeds the bounds of the signal stack.
        signal_stack[i] = \
            add_array_with_interpolation(array_large=signal_stack[i],
                                         array_small=psf_cropped,
                                         position=injection_position)

    return signal_stack, planet_positions
