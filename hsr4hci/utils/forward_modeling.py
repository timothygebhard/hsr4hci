"""
Utility functions for forward modeling.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Dict, List, Tuple, Union

import math

from scipy.interpolate import RegularGridInterpolator

import numpy as np

from hsr4hci.utils.general import add_array_with_interpolation, rotate_position
from hsr4hci.utils.utils import check_frame_size, check_cartesian_position


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def compute_signal_stack(
    position: Tuple[float, float],
    frame_size: Tuple[int, int],
    parang: np.ndarray,
    psf_template: np.ndarray,
    return_planet_positions: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute a "forward model": Given the parallactic angles in `parang`
    and the PSF template in `psf_cropped`, create a "signal only"-stack
    of shape `(n_frames, width, height)` where in each frame a signal is
    injected such that, if the whole stack is de-rotated and combined,
    we find the planet signal at the target `position`.

    Args:
        position: A tuple `(x, y)` containing the desired "final" (i.e.,
            after de-rotating) position of the planet.
            Coordinate convention:
                system: Cartesian, origin: astropy, order: numpy
        frame_size: A 2-tuple `(width, height)` of integers specifying
            the spatial size of the signal stack to be created.
        parang: A numpy array of shape `(n_frames, )` containing the
            parallactic angle for every frame.
        psf_template: The (normalized and cropped) PSF template that is
            used for the signal
        return_planet_positions: Whether or not to return a 2D numpy
            array of shape `(n_frames, 2)` containing planet position
            for each frame.

    Returns:
        A tuple consisting of a numpy array containing the forward model
        (i.e., a stack of frames where every frame contains only the
        "planet signal" at the correct position), computed under the
        given assumptions, as well as a list of tuples (x, y) containing
        the position of the planet in the forward model for every frame.
    """

    # Run basic sanity checks
    check_frame_size(frame_size)
    check_cartesian_position(position)

    # Define some shortcuts
    n_frames = len(parang)
    center = (frame_size[0] / 2, frame_size[1] / 2)

    # Convert parang from degrees to radians
    parang = np.deg2rad(parang)

    # Compute polar representation of initial position
    x_centered = position[0] - center[0]
    y_centered = position[1] - center[1]
    rho = math.sqrt(x_centered ** 2 + y_centered ** 2)
    phi = math.atan2(y_centered, x_centered)

    # Compute planet positions for each frame
    x_positions = rho * np.cos(phi - parang) + center[0]
    y_positions = rho * np.sin(phi - parang) + center[1]

    # Initialize empty signal stack and list of planet positions
    signal_stack = np.zeros((n_frames, frame_size[0], frame_size[1]))
    planet_positions = np.column_stack((x_positions, y_positions))

    # Loop over all frames and add the cropped and masked PSF template at
    # the position that corresponds to the respective parallactic angle
    for i, position in zip(np.arange(n_frames), planet_positions):

        # Add cropped and masked PSF template at the injection_position.
        # The add_array_with_interpolation() function is able to automatically
        # deal with cases where the injection_position is a tuple of floats,
        # or the PSF template exceeds the bounds of the signal stack.
        signal_stack[i] = add_array_with_interpolation(
            array_large=signal_stack[i],
            array_small=psf_template,
            position=position[::-1],
        )

    # Only return planet positions if explicitly requested
    if return_planet_positions:
        return signal_stack, planet_positions
    return signal_stack


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

    # Run basic sanity checks
    check_frame_size(frame_size)
    check_cartesian_position(position)

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
        planet_model, planet_positions = compute_signal_stack(
            position=starting_position,
            frame_size=(x_size, y_size),
            parang=parang,
            psf_template=psf_cropped,
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


def add_fake_planet(
    stack: np.ndarray,
    parang: np.ndarray,
    psf_template: np.ndarray,
    position: Tuple[float, float],
    magnitude: float,
    extra_scaling: float,
    dit_stack: float,
    dit_psf_template: float,
    return_planet_positions: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List[Tuple[float, float]]]]:
    """
    Add a fake planet to the given `stack` which, when derotating and
    merging the stack, will show up at the given `position`.

    This function can also be used to *remove* planets from a stack by
    setting the `psf_scaling` to a negative number.

    Args:
        stack: A 3D numpy array of shape `(n_frames, width, height)`
            which contains the stack of images / frames into which we
            want to inject a fake planet.
        parang: A 1D numpy array of shape `(n_frames, )` that contains
            the respective parallactic angle for every frame in `stack`.
        psf_template: A 2D numpy array that contains the (centered) PSF
            template which will be used for the fake planet.
            This should *not* be normalized to (0, 1] if we want to work
            with actual astrophysical magnitudes for the contrast.
        position: A tuple `(x, y)` which specifies the position at which
            the planet will show up after de-rotating with `parang`.
        magnitude: The magnitude difference used to scale the PSF.
            Note: This is the contrast ratio in *magnitudes*, meaning
            that increasing this value by a factor of 5 will result in
            a planet that is 100 times brighter. In case you want to
            keep things linear, set this value to 0 and only use the
            `psf_scaling` parameter.
        extra_scaling: An additional scaling factor that is used for
            the PSF template.
            This number is simply multiplied with the PSF template,
            meaning that it changes the brightness linearly, not on a
            logarithmic scale. For example, you could use `-1` to add a
            *negative* planet to remove an actual planet in the data.
            This can also be used to incorporate an additional dimming
            factor due to a
        dit_stack: The detector integration time of the frames in the
            `stack` (in seconds). Necessary to compute the correct
            scaling factor for the planet that we inject.
        dit_psf_template: The detector integration time of the
            `psf_template` (in seconds). Necessary to compute the
            correct scaling factor for the planet that we inject.
        return_planet_positions:

    Returns:
        A 3D numpy array of shape `(n_frames, width, height)` which
        contains the original `stack` into which a fake planet has been
        injected, as well as a list of tuples `(x, y)` that, for each
        frame, contain the position at which the fake planet has been
        added.
    """

    # Define some shortcuts
    frame_size = stack.shape[1:]

    # Convert `magnitude` from logarithmic contrast to linear flux ratio
    contrast_scaling = 10 ** (-magnitude / 2.5)

    # Compute scaling factor that is due to the different integration times
    # for the science images and the PSF template
    dit_scaling = dit_stack / dit_psf_template

    # Combine all scaling factors and scale the PSF template
    scaling_factor = contrast_scaling * dit_scaling * extra_scaling
    psf_scaled = scaling_factor * psf_template

    # Compute a stack
    planet_stack, planet_positions = compute_signal_stack(
        position=position,
        frame_size=frame_size,
        parang=parang,
        psf_template=psf_scaled,
        return_planet_positions=True,
    )

    # Add the planet stack to the original input stack
    output_stack = stack + planet_stack

    if return_planet_positions:
        return output_stack, planet_positions
    return output_stack
