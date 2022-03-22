"""
Methods for forward modeling (= simulating a pure planet signal).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import List, Tuple, Union, overload
from typing_extensions import Literal

from astropy.units import Quantity
from scipy.interpolate import RegularGridInterpolator

import numpy as np

from hsr4hci.coordinates import cartesian2polar, get_center
from hsr4hci.general import (
    crop_or_pad,
    rotate_position,
    shift_image,
)
from hsr4hci.utils import (
    check_consistent_size,
    check_frame_size,
    check_cartesian_position,
)


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

@overload
def add_fake_planet(
    stack: np.ndarray,
    parang: np.ndarray,
    psf_template: np.ndarray,
    polar_position: Tuple[Quantity, Quantity],
    magnitude: float,
    extra_scaling: float,
    dit_stack: float,
    dit_psf_template: float,
    return_planet_positions: Literal[True],
    interpolation: str = 'bilinear',
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    ...  # pragma: no cover


@overload
def add_fake_planet(
    stack: np.ndarray,
    parang: np.ndarray,
    psf_template: np.ndarray,
    polar_position: Tuple[Quantity, Quantity],
    magnitude: float,
    extra_scaling: float,
    dit_stack: float,
    dit_psf_template: float,
    return_planet_positions: Literal[False],
    interpolation: str = 'bilinear',
) -> np.ndarray:
    ...  # pragma: no cover


def add_fake_planet(
    stack: np.ndarray,
    parang: np.ndarray,
    psf_template: np.ndarray,
    polar_position: Tuple[Quantity, Quantity],
    magnitude: float,
    extra_scaling: float,
    dit_stack: float,
    dit_psf_template: float,
    return_planet_positions: bool = False,
    interpolation: str = 'bilinear',
) -> Union[np.ndarray, Tuple[np.ndarray, List[Tuple[float, float]]]]:
    """
    Add a fake planet to the given `stack` which, when derotating and
    merging the stack, will show up at the given `position`.

    This function can also be used to *remove* planets from a stack by
    setting the `psf_scaling` to a negative number.

    If you simply want to use this function to generate a fake signal
    stack, set `stack` to all zeros, the `magnitude` to zero, both the
    `dit_stack` and `dit_psf_template` to 1 (or any other non-zero
    number), and use the `extra_scaling` factor to linearly control
    the brightness of the injected planet.

    This function is essentially a simplified port of the corresponding
    PynPoint function `pynpoint.util.analysis.fake_planet()`.

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
        polar_position: A tuple `(separation, angle)` which specifies
            the position at which the planet will show up after
            de-rotating with `parang`. `separation` needs to be a
            Quantity that can be converted to pixel; `angle` needs to
            be a Quantity that can be converted to radian. Additionally,
            `angle` should be using *astronomical* polar coordinates,
            that is, 0 degrees will be "up" (= North), not "right".
            This function will internally add 90° to the angles to
            convert them to mathematical pilar coordinates.
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
        return_planet_positions: Whether to return the (Cartesian)
            positions at which the fake planet was injected, as a
            2D numpy array of shape `(n_frames, 2)`.
        interpolation: interpolation argument that is passed to the
            `scipy.ndimage.shift()` routine that is used internally.

    Returns:
        A 3D numpy array of shape `(n_frames, width, height)` which
        contains the original `stack` into which a fake planet has been
        injected, as well as a list of tuples `(x, y)` that, for each
        frame, contain the position at which the fake planet has been
        added.
        If desired, additionally also a 2D numpy array of shape
        `(n_frames, 2)` containing the Cartesian positions at which
        the fake planet has been injected.
    """

    # Make sure that the stack and the parallactic angles are compatible
    check_consistent_size(stack, parang)

    # Define shortcut for the number of frames and the frame_size
    n_frames, frame_size = stack.shape[0], (stack.shape[1], stack.shape[2])

    # Split the target planet position into separation and angles, convert
    # the quantities to pixels / convert to mathematical polar coordinates
    rho = polar_position[0].to('pixel').value
    phi = np.radians(polar_position[1].to('degree').value + 90 - parang)

    # Convert `magnitude` from logarithmic contrast to linear flux ratio
    flux_ratio = 10.0 ** (-magnitude / 2.5)

    # Compute scaling factor that is due to the different integration times
    # for the science images and the PSF template
    dit_scaling = dit_stack / dit_psf_template

    # Combine all scaling factors and scale the PSF template
    scaling_factor = flux_ratio * dit_scaling * extra_scaling
    psf_scaled = scaling_factor * np.copy(psf_template)

    # Make sure that the PSF has a compatible shape, that is, either crop or
    # pad the PSF template to the same spatial shape as the `stack`.
    psf_scaled = crop_or_pad(psf_scaled, frame_size)

    # Compute the shift for each frame
    x_shift = rho * np.cos(phi)
    y_shift = rho * np.sin(phi)

    # Initialize the "pure signal" stack (can use empty() here, because all
    # values will be overwritten and allocation should be slightly faster)
    signal_stack = np.empty_like(stack)

    # For each frame, move the scaled PSF template to the correct position
    # Note: We use mode='constant' instead of 'reflect' here (unlike PynPoint)
    # because the latter just does not seem to make a lot of sense?
    for i in range(n_frames):
        signal_stack[i] = shift_image(
            image=psf_scaled,
            offset=(float(x_shift[i]), float(y_shift[i])),
            interpolation=interpolation,
            mode='constant',
        )

    # Add the planet stack to the original input stack
    output_stack = stack + signal_stack

    # Either return only the output stack, or the output stack and
    # the planet positions
    if return_planet_positions:
        center = get_center(frame_size)
        planet_positions = np.column_stack(
            (x_shift + center[0], y_shift + center[1])
        )
        return output_stack, planet_positions
    return np.array(output_stack)


def get_time_series_for_position(
    position: Tuple[float, float],
    signal_time: int,
    frame_size: Tuple[int, int],
    parang: np.ndarray,
    psf_template: np.ndarray,
    interpolation: str = 'bilinear',
) -> np.ndarray:
    """
    Compute the expected signal time series for a pixel at `position`
    under the assumption that the planet signal is centered on this
    pixel at the given `signal_time`.

    If we are only interested in a single such time series, using this
    function will be *dramatically* faster than computing the full stack
    using `get_signal_stack()` and selecting the position of interest.

    The idea behind this function is that we can get the time series of
    interest (or a reasonably good approximation of it) by creating a
    single frame of all zeros, into which we place the PSF template at
    the target `position`, and then sample this array along the implied
    path (determined by the fact that the signal is supposed to be at
    `position` at the `signal_time`) of the planet. This avoids the
    computationally expensive generation of the full stack (of which
    only a single spatial pixel would be used). The exact speed-up of
    this version depends on the number of frames, but is typically on
    the order of a factor of 10^2 to 10^3.

    Note: This function uses the *numpy convention* for `position`!

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
        psf_template: A numpy array containing the cropped and masked
            PSF template, as it is returned by `crop_psf_template()`.
        interpolation: interpolation parameter that is passed to the
            `shift_image()` function. Default is "bilinear".

    Returns:
        The time series for `position` computed under the hypothesis for
        the planet movement explained above.
    """

    # Run basic sanity checks
    check_frame_size(frame_size)
    check_cartesian_position(position)

    # Compute center of the frame
    center = get_center(frame_size=frame_size)

    # Make sure that the PSF has a compatible shape, that is, either crop or
    # pad the PSF template to the same shape as the `stack`.
    psf_cropped = np.copy(psf_template)
    psf_cropped = crop_or_pad(psf_cropped, frame_size)

    # Create array where we place the PSF template at the target `position`
    array = shift_image(
        image=psf_cropped,
        offset=(position[0] - center[0], position[1] - center[1]),
        interpolation=interpolation,
    )

    # Find the starting position of the planet under the hypothesis given by
    # `position` and `signal_time`
    starting_position = rotate_position(
        position=position, angle=-parang[signal_time], center=center
    )

    # Compute the full array of all planet positions (at all times)
    planet_positions = np.asarray(
        rotate_position(
            position=starting_position,
            angle=parang,
            center=center,
        )
    ).T

    # Create an interpolator for the array that allows us to evaluate it also
    # at non-integer positions. This function uses (bi)-linear interpolation.
    x_range = np.arange(frame_size[0])
    y_range = np.arange(frame_size[1])
    interpolator = RegularGridInterpolator((x_range, y_range), array)

    # The target time series is given by (interpolated) array values at the
    # positions along the planet path. Note that we need to flip the order of
    # the planet positions because we are basically accessing a numpy array.
    time_series = np.array(interpolator(planet_positions[:, ::-1]))

    # Make sure that the time series is normalized to a maximum of 1
    time_series /= np.nanmax(time_series)

    return time_series


def get_time_series_for_position__full_stack(
    position: Tuple[float, float],
    signal_time: int,
    frame_size: Tuple[int, int],
    parang: np.ndarray,
    psf_template: np.ndarray,
    interpolation: str = 'spline',
) -> np.ndarray:
    """
    This function does the same as `get_time_series_for_position()`, but
    it does not use our trick to speed up the computation; instead, the
    *full* signal stack is generated. This function should only ever be
    used to verify the correctness of `get_time_series_for_position()`,
    as it will probably be too slow for most practical applications.
    """

    # Run basic sanity checks
    check_frame_size(frame_size)
    check_cartesian_position(position)

    # Compute center of the frame
    n_frames = len(parang)
    center = get_center(frame_size=frame_size)

    # Make sure that the PSF has a compatible shape, that is, either crop or
    # pad the PSF template to the same shape as the `stack`.
    psf_cropped = np.copy(psf_template)
    psf_cropped = crop_or_pad(psf_cropped, frame_size)

    # Given that the hypothesis that the planet is at `position` at the given
    # `signal_time`, compute the planet position in the final images
    final_position_cartesian = rotate_position(
        position=position, angle=parang[signal_time], center=center
    )
    final_position_polar = cartesian2polar(
        position=(final_position_cartesian[0], final_position_cartesian[1]),
        frame_size=frame_size,
    )

    # Compute the full signal stack
    signal_stack = add_fake_planet(
        stack=np.zeros((n_frames,) + frame_size),
        parang=parang,
        psf_template=psf_cropped,
        polar_position=final_position_polar,
        magnitude=0,
        extra_scaling=1,
        dit_stack=1,
        dit_psf_template=1,
        return_planet_positions=False,
        interpolation=interpolation,
    )

    # Create an interpolator for the array that allows us to evaluate it also
    # at non-integer positions. This function uses (bi)-linear interpolation.
    t_range = np.arange(n_frames)
    x_range = np.arange(frame_size[0])
    y_range = np.arange(frame_size[1])
    interpolator = RegularGridInterpolator(
        points=(t_range, x_range, y_range), values=signal_stack
    )

    # Get the stack values for the given target position using the interpolator
    # Note that we need to flip the order of position because we are basically
    # accessing a numpy array here.
    dummy = np.array([[t, position[1], position[0]] for t in t_range])
    time_series = np.array(interpolator(dummy))

    # Make sure that the time series is normalized to a maximum of 1
    time_series /= np.nanmax(time_series)

    return time_series
