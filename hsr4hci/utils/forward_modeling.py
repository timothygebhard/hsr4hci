"""
Utility functions for forward modeling (necessary for toy data sets!).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from cmath import polar
from typing import Dict, Tuple

from scipy.interpolate import RegularGridInterpolator

import numpy as np

from hsr4hci.utils.general import add_array_with_interpolation, rotate_position


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
    r, phi = polar(
        complex(position[1] - frame_center[1], position[0] - frame_center[0])
    )

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
        injection_position = (
            np.imag(new_complex_position) + frame_center[0],
            np.real(new_complex_position) + frame_center[1],
        )
        planet_positions.append(injection_position)

        # Add cropped and masked PSF template at the injection_position.
        # The add_array_with_interpolation() function is able to automatically
        # deal with cases where the injection_position is a tuple of floats,
        # or the PSF template exceeds the bounds of the signal stack.
        signal_stack[i] = add_array_with_interpolation(
            array_large=signal_stack[i],
            array_small=psf_cropped,
            position=injection_position,
        )

    return signal_stack, planet_positions


def get_time_series_for_position(
    position: Tuple[float, float],
    signal_time: int,
    frame_size: Tuple[int, int],
    parang: np.ndarray,
    psf_cropped: np.ndarray,
) -> np.ndarray:
    """
    Compute the expected signal time series for a pixel at `position`
    under the assumption that the planet signal is centered on this
    pixel at the given `signal_time`.

    If we are only interested in a single such time series, using this
    function will be *dramatically* faster than computing the full stack
    using `get_signal_stack()` and selecting the position of interest.

    The idea behind this function is that we can get the time series of
    interest (or a very good approximation of it) by creating a single
    frame of all zeros, into which we place the PSF template at the
    target `position`, and then sample this array along the implied
    path (determined by the fact that the signal is supposed to be at
    `position` at the `signal_time`) of the planet.

    Args:
        position: A tuple `(x, y)` for which we want to compute the
            time series under a given planet path hypothesis.
        signal_time: An integer specifying the time (= the frame number)
            at which the signal is to be assumed to be centered on the
            pixel at the given `position`.
        frame_size: A tuple `(x_size, y_size)` giving the spatial size
            of the stack.
        parang: A numpy array containing the parallactic angle for
            every frame.
        psf_cropped: A numpy array containing the cropped and masked
            PSF template, as it is returned by `crop_psf_template()`.

    Returns:
        The time series for `position` computed under the hypothesis for
        the planet movement explained above.
    """

    # Compute center of the frame
    center = (frame_size[0] / 2, frame_size[1] / 2)

    # Create array where we place the PSF template at the target `position`
    array = add_array_with_interpolation(
        array_large=np.zeros(frame_size),
        array_small=psf_cropped,
        position=position,
    )

    # Find the starting position of the planet under the hypothesis
    angle = parang[0] - parang[signal_time]
    starting_position = rotate_position(
        position=position, angle=angle, center=center
    )

    # Compute the full array of all planet positions (at all times)
    planet_positions = np.vstack(
        rotate_position(
            position=starting_position,
            angle=(parang - parang[0]),
            center=center,
        )
    ).T

    # Create an interpolator for the array that allows us to evaluate it also
    # at non-integer positions. This function uses (bi)-linear interpolation.
    x_range = np.arange(frame_size[0])
    y_range = np.arange(frame_size[1])
    interpolator = RegularGridInterpolator((x_range, y_range), array)

    # The target time series is given by (interpolated) array values at the
    # positions along the planet path
    time_series = interpolator(planet_positions)

    # Make sure that the time series is normalized to a maximum of 1
    time_series /= np.nanmax(time_series)

    return time_series


def get_planet_paths(
    stack_shape: Tuple[int, int, int],
    parang: np.ndarray,
    psf_cropped: np.ndarray,
    planet_config: Dict[str, dict],
    threshold: float = 5e-1,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Auxiliary function for computing the paths of the planets in the
    data set (as a list of tuples indicating positions), as well as for
    creating binary masks that indicate which spatial pixels in a data
    set contain planet signal at some point in time.

    Args:
        stack_shape: A tuple `(n_frames, x_size, y_size)` giving the
            shape of the stack of the data set.
        parang: A 1D numpy array of shape `(n_frames, )` containing the
            parallactic angles.
        psf_cropped: A 2D numpy array containing a cropped version of
            the raw, unsaturated PSF template.
        planet_config: A dictionary containing information about the
            planets in the data set. Each key (usually the letter that
            indicates the name of the planet, e.g., "b") must map onto
            another dictionary, which contains a key named "position",
            which specifies the position of the planet in the *final*,
            derotated signal estimate (e.g., the PCA estimate).
        threshold: The threshold value for creating the binary mask;
            basically the minimum planet signal to be contained in a
            pixel (at some point) to add this pixel to the mask.

    Returns:
        A 2D numpy array of shape `(x_size, y_size)` in which every
        pixel that at some point in time contains planet signal is True
        and all other pixels are False.
    """

    # Define shortcuts
    n_frames, x_size, y_size = stack_shape
    center = (x_size / 2, y_size / 2)

    # Instantiate an empty stack-like variable from which we will generate the
    # mask, and a dictionary which will hold the planet positions
    stack = np.zeros(stack_shape)
    all_planet_positions: Dict[str, np.ndarray] = dict()

    # Loop over the different planets for the data set
    for key, values in planet_config.items():

        # Get final position of the planet
        final_position = values['position'][::-1]

        # Compute the starting position of the planet
        starting_position = rotate_position(
            position=final_position,
            center=center,
            angle=parang[0],
        )

        # Compute a forward model for this planet
        planet_model, planet_positions = get_signal_stack(
            position=starting_position,
            frame_size=(x_size, y_size),
            parang=parang,
            psf_cropped=psf_cropped,
        )

        # Convert the planet positions to a 2D array and store them
        planet_positions = np.array(planet_positions)
        all_planet_positions[key] = planet_positions

        # Add this to the existing stack
        stack += planet_model

    # Compute the mask of the planet paths by taking the maximum along the
    # temporal dimension and thresholding the result
    planet_paths_mask = np.max(stack, axis=0) > threshold

    return planet_paths_mask, all_planet_positions
