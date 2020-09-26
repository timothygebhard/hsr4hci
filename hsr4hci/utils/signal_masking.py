"""
Utility functions for signal masking and related tasks.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from cmath import polar
from typing import List, Tuple

import numpy as np

from hsr4hci.utils.forward_modeling import get_time_series_for_position


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_effective_pixel_width(
    position: Tuple[float, float],
    center: Tuple[float, float],
) -> float:
    """
    Compute the "effective" width of a pixel, that is, the the length of
    the path of a planet that crosses the center of the given pixel.

    This value will be between 1 and sqrt(2), depending on the position
    of the pixel (namely, it is a function of the polar angle).

    Args:
        position: A tuple (x, y) specifying the position of a pixel.
        center: A tuple (c_x, c_y) specifying the frame center.

    Returns:
        A value from [0, sqrt(2)] that is the "effective" pixel width.
    """

    # Get polar angle and make sure it is in [0, pi / 2], so that we do not
    # have to distinguish between different quadrants
    _, phi = polar(complex(position[1] - center[1], position[0] - center[0]))
    _, phi = divmod(phi, np.pi / 2)

    # Compute the effective pixel width, which is effectively either the
    # secans (= 1/cos) or cosecans (= 1/sin) of the polar angle
    effective_pixel_width = min(float(1 / np.cos(phi)), float(1 / np.sin(phi)))

    return effective_pixel_width


def get_signal_length(
    position: Tuple[float, float],
    signal_time: int,
    center: Tuple[float, float],
    parang: np.ndarray,
    psf_diameter: float,
) -> Tuple[int, int]:
    """
    Get a simple analytical estimate of the length (in units of frames)
    that a planet (with a PSF of the given `psf_diameter`) passing
    through a pixel at the given `position` and time `frame_idx` will
    produce, that is, the number of (consecutive) frames that will
    contain planet signal.

    Taking the `signal_time` into account is necessary because the
    temporal derivative of the parallactic angle is non-constant over
    the course of an observation, but can change up to around 50% for
    some data sets.

    Args:
        position:
        signal_time:
        center:
        parang:
        psf_diameter:

    Returns:

    """

    # Check if the parallactic angles are sorted in ascending order
    if np.allclose(parang, sorted(parang)):
        ascending = True
    elif np.allclose(parang, sorted(parang, reverse=True)):
        ascending = False
    else:
        raise ValueError('parang is not sorted!')

    # Compute radius of position
    radius = np.sqrt(
        (position[0] - center[0]) ** 2 + (position[1] - center[1]) ** 2
    )

    # Compute the effective pixel width of the position
    effective_pixel_width = get_effective_pixel_width(
        position=position, center=center
    )

    # Convert "effective pixel width + PSF diameter" to an angle at this radius
    # using the cosine theorem. First, compute the length of the side that we
    # want to convert into an angle:
    side_length = effective_pixel_width + psf_diameter

    # Degenerate case: for too small separations, if the center is ever on the
    # pixel, the pixel will *always* contain planet signal.
    if side_length > 2 * radius:
        return 0, len(parang)

    # Otherwise, we can convert the side length into an angle
    gamma = np.arccos(1 - side_length ** 2 / (2 * radius ** 2))

    # Find positions
    value_1 = parang[signal_time] - np.rad2deg(gamma) / 2
    value_2 = parang[signal_time] + np.rad2deg(gamma) / 2
    if ascending:
        position_1 = np.searchsorted(parang, value_1, side='left')
        position_2 = np.searchsorted(parang, value_2, side='right')
    else:
        position_2 = np.searchsorted(-parang, -value_1, side='left')
        position_1 = np.searchsorted(-parang, -value_2, side='right')

    # Compute the length before and after the peak (because the signal will, in
    # general, not be symmetric around the peak)
    length_1 = int(1.0 * (signal_time - position_1))
    length_2 = int(1.0 * (position_2 - signal_time))

    return length_1, length_2


def get_signal_masks_analytically(
    position: Tuple[int, int],
    parang: np.ndarray,
    n_signal_times: int,
    frame_size: Tuple[int, int],
    psf_diameter: float,
    max_signal_length: float = 0.7,
) -> List[Tuple[int, np.ndarray, int]]:
    """
    Generate the masks for training a series of models where different
    possible planet positions are masked out during training.

    This function places `n_signal_times` points in time uniformly over
    the course of the whole observation. For each such time point, we
    then assume that the planet signal is present at this point in time,
    and generate a binary mask that indicates all points in time that,
    under this hypothesis, would also contain planet signal.

    This function uses an "analytical" approximation based on
    `get_effective_pixel_width()` and `get_signal_length()` to estimate
    the signal masks, which is fast, but not as accurate as really
    modeling the movement of the planet signal in time.

    Args:
        position: An integer tuple `(x, y)` specifying the spatial
            position of the pixel for which we are computing the masks.
        parang: A numpy array of shape `(n_frames, )` containing the
            parallactic angles.
        n_signal_times: The number of different possible temporal
            positions of the planet signal for which to return a mask.
        frame_size: A tuple `(width, height)` specifying the spatial
            size of the stack.
        psf_diameter: The diameter of the PSF template (in pixels).
        max_signal_length: A value in [0.0, 1.0] which describes the
            maximum value of `expected_signal_length / n_frames`, which
            will determine for which pixels we do not want to use the
            "mask out a potential signal region"-approach, because the
            potential signal region is too large to leave us with a
            reasonable amount of training data.

    Returns:
        This function returns a list of up to `n_position` 3-tuples
        of the following form:
            `(signal_time_index, signal_mask, signal_time)`.
    """

    # Define shortcuts
    n_frames = len(parang)
    center = (frame_size[0] / 2, frame_size[1] / 2)

    # Initialize lists in which we store the results
    results = list()

    # Generate `n_signal_times` different possible points in time (distributed
    # uniformly over the observation) at which we planet signal could be
    signal_times = np.linspace(0, n_frames - 1, n_signal_times)

    # Loop over all these time points to generate the corresponding indices
    for i, signal_time in enumerate(signal_times):

        # Make sure the signal time is an integer (we use it as an index)
        signal_time = int(signal_time)

        # Compute the expected signal length at this position and time
        length_1, length_2 = get_signal_length(
            position=position,
            signal_time=signal_time,
            center=center,
            parang=parang,
            psf_diameter=psf_diameter,
        )

        # Check if the expected signal length is larger than the threshold.
        # In this case, we do not compute the noise and signal masks, but
        # skip this signal time.
        if (length_1 + length_2) / n_frames > max_signal_length:
            continue

        # Construct the signal mask
        signal_mask = np.full(n_frames, False)
        position_1 = max(0, signal_time - length_1)
        position_2 = min(n_frames, signal_time + length_2)
        signal_mask[position_1:signal_time] = True
        signal_mask[signal_time:position_2] = True

        # Store the current (signal_time_index, signal_mask, signal_time) tuple
        results.append((i, signal_mask, signal_time))

    return results


def get_signal_masks(
    position: Tuple[int, int],
    parang: np.ndarray,
    n_signal_times: int,
    frame_size: Tuple[int, int],
    psf_cropped: np.ndarray,
    max_signal_length: float = 0.7,
) -> List[Tuple[int, np.ndarray, int]]:
    """
    Generate the masks for training a series of models where different
    possible planet positions are masked out during training.

    Similar to `get_signal_masks_analytically()`; however, this version
    makes use of `get_time_series_for_position()` to accurately model
    the expected shape of the planet signal at the given `position`, and
    determines the signal mask by thresholding this expected signal.

    Args:
        position: An integer tuple `(x, y)` specifying the spatial
            position of the pixel for which we are computing the masks.
        parang: A numpy array of shape `(n_frames, )` containing the
            parallactic angles.
        n_signal_times: The number of different possible temporal
            positions of the planet signal for which to return a mask.
        frame_size: A tuple `(width, height)` specifying the spatial
            size of the stack.
        psf_cropped: A 2D numpy array containing a (cropped) version of
            the PSF template. Typically, the PSF is cropped to a radius
            of 1 lambda over D (to only contain the central peak) or
            3 lambda over D (to also capture the secondary maxima).
        max_signal_length: A value in [0.0, 1.0] which describes the
            maximum value of `expected_signal_length / n_frames`, which
            will determine for which pixels we do not want to use the
            "mask out a potential signal region"-approach, because the
            potential signal region is too large to leave us with a
            reasonable amount of training data.

    Returns:
        This function returns a list of up to `n_position` 3-tuples
        of the following form:
            `(signal_time_index, signal_mask, signal_time)`.
    """

    # Define shortcuts
    n_frames = len(parang)

    # Initialize lists in which we store the results
    results = list()

    # Generate `n_signal_times` different possible points in time (distributed
    # uniformly over the observation) at which we planet signal could be
    signal_times = np.linspace(0, n_frames - 1, n_signal_times)

    # Loop over all these time points to generate the corresponding indices
    for i, signal_time in enumerate(signal_times):

        # Make sure the signal time is an integer (we use it as an index)
        signal_time = int(signal_time)

        # Compute the expected time series for given `position` under the
        # assumption that a planet is there at the current `signal_time`
        expected_signal = get_time_series_for_position(
            position=position,
            signal_time=signal_time,
            frame_size=frame_size,
            parang=parang,
            psf_cropped=psf_cropped,
        )

        # Threshold the expected signal to create a binary mask.
        # The threshold value here is somewhat arbitrary (?)
        signal_mask = expected_signal > 0.2

        # Check if the expected signal length is larger than the threshold.
        # In this case, we do not compute the noise and signal masks, but
        # skip this signal time.
        if np.sum(signal_mask) / n_frames > max_signal_length:
            continue

        # Store the current (signal_time_index, signal_mask, signal_time) tuple
        results.append((i, signal_mask, signal_time))

    return results
