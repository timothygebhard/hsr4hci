"""
Utility functions for signal masking and related tasks.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from itertools import product
from typing import Dict, List, Tuple, Union

import numpy as np

from hsr4hci.utils.forward_modeling import get_time_series_for_position


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def assemble_signal_masking_residuals(
    hypotheses: np.ndarray,
    results: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]],
) -> np.ndarray:
    """
    Assemble the signal masking results based on the `hypotheses`, that
    is, for each spatial pixel use the residual that was obtained when
    assuming the signal time that equals the hypothesis for this pixel.

    Args:
        hypotheses: A 2D numpy array of shape `(width, height)`. Each
            position contains an integers which represents the signal
            time which appears to be the best hypothesis for this pixel
            (i.e., "if there ever is a planet in this pixel, it should
            be at this time").
        results: The dictionary which contains the full results from
            training both the "default" and the signal masking based
            models.

    Returns:
        A 3D numpy array (whose shape matches the stack) containing the
        "best" signal masking-based residuals given the `hypotheses`.
    """

    # Get stack shape from default residuals
    n_frames, x_size, y_size = results['default']['residuals'].shape

    # Initialize the signal masking residuals as all-NaN
    signal_masking_residuals = np.full((n_frames, x_size, y_size), np.nan)

    # Loop over all spatial positions and pick the signal masking-based
    # residual based on the the respective hypothesis for the pixel
    for x, y in product(range(x_size), range(y_size)):
        if not np.isnan(signal_time := hypotheses[x, y]):
            signal_masking_residuals[:, x, y] = np.array(
                results[str(int(signal_time))]['residuals'][:, x, y]
            )

    return signal_masking_residuals


def get_signal_length(
    position: Tuple[float, float],
    signal_time: int,
    frame_size: Tuple[int, int],
    parang: np.ndarray,
    psf_template: np.ndarray,
    threshold: float = 0.2,
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
        frame_size:
        parang:
        psf_template:
        threshold:

    Returns:

    """

    # Compute the expected time series
    expected_time_series = get_time_series_for_position(
        position=position,
        signal_time=signal_time,
        frame_size=frame_size,
        parang=parang,
        psf_template=psf_template,
        interpolation='bilinear',
    )

    # Threshold it
    signal_mask = expected_time_series > threshold

    # Find index of first and last 1 on this binary array
    indices = np.arange(len(signal_mask))[signal_mask]
    position_1 = np.min(indices)
    position_2 = np.max(indices)

    # Compute the length before and after the peak (because the signal will, in
    # general, not be symmetric around the peak)
    before = signal_time - position_1
    after = position_2 - signal_time

    return before, after


def get_signal_mask(
    position: Tuple[float, float],
    parang: np.ndarray,
    signal_time: int,
    frame_size: Tuple[int, int],
    psf_template: np.ndarray,
    threshold: float = 0.2,
) -> np.ndarray:
    """
    Get the signal mask for a single spatio-temporal planet position
    given by the `position` and the `signal_time`.

    This function only returns a single mask, corresponding to a single
    hypothesis regarding the planet's trajectory!

    Args:
        position: A tuple `(x, y)` specifying the position at which the
            planet is assumed to be at time `signal_time`.
        parang: A 1D numpy array of shape `(n_frames, )` containing the
            parallactic angles for each frame.
        signal_time: An integer specifying the index of the frame in
            which the planet is assumed to be at the given `position`.
        frame_size: A tuple `(width, height)` specifying the size (in
            pixels) of the frames that we are working with.
        psf_template: A 2D numpy array containing a (cropped( version of
            the unsaturated PSF template for the data set. In essence,
            this defines the (spatial) size of the planet signal on the
            sensor of the instrument.
        threshold: The threshold that is used when binarizing the
            expected planet signal into a mask. The exact value is
            probably somewhat arbitrary. Smaller values give larger
            masks, which means that more data are excluded during
            training. Default is 0.2 = 20% of the maximum signal.

    Returns:
        A binary 1D numpy array of shape `(n_frames, )` that is True at
        all times (indices) where the target pixel does contain planet
        signal (under the hypothesis that the planet actually is a
        `position`  at the given `signal_time`), and False elsewhere.
    """

    # Make sure the signal time is an integer (we use it as an index)
    signal_time = int(signal_time)

    # Compute the expected time series for given `position` under the
    # assumption that a planet is there at the current `signal_time`
    expected_signal = get_time_series_for_position(
        position=position,
        signal_time=signal_time,
        frame_size=frame_size,
        parang=parang,
        psf_template=psf_template,
    )

    # Threshold the expected signal to create a binary mask
    signal_mask = np.asarray(expected_signal > threshold)

    return signal_mask


def get_signal_masks(
    position: Tuple[int, int],
    parang: np.ndarray,
    signal_times: np.ndarray,
    frame_size: Tuple[int, int],
    psf_template: np.ndarray,
    max_signal_length: float = 0.7,
    threshold: float = 0.2,
) -> List[Tuple[np.ndarray, int]]:
    """
    Generate the masks for training a series of models where different
    possible signal times are masked out during training.

    This function returns a *list of masks* (and signal times); one for
    each signal time on the temporal grid that is used for training the
    HSR models with signal masking! It is basically a wrapper around
    `get_signal_mask()` that runs the loop over the `signal_times` grid.

    Similar to `get_signal_masks_analytically()`; however, this version
    makes use of `get_time_series_for_position()` to accurately model
    the expected shape of the planet signal at the given `position`, and
    determines the signal mask by thresholding this expected signal.

    Args:
        position: An integer tuple `(x, y)` specifying the spatial
            position of the pixel for which we are computing the masks.
        parang: A numpy array of shape `(n_frames, )` containing the
            parallactic angles.
        signal_times: A 1D numpy array containing the temporal grid
            of possible signal times for which to return a mask.
        frame_size: A tuple `(width, height)` specifying the spatial
            size of the stack.
        psf_template: A 2D numpy array containing a (cropped) version of
            the PSF template. Typically, the PSF is cropped to a radius
            of 1 lambda over D (to only contain the central peak) or
            3 lambda over D (to also capture the secondary maxima).
        max_signal_length: A value in [0.0, 1.0] which describes the
            maximum value of `expected_signal_length / n_frames`, which
            will determine for which pixels we do not want to use the
            "mask out a potential signal region"-approach, because the
            potential signal region is too large to leave us with a
            reasonable amount of training data.
        threshold: The threshold value that is passed to the
            `get_signal_mask` function (see there for details).

    Returns:
        This function returns a list of up to `n_position` 3-tuples
        of the following form: `(signal_mask, signal_time)`.
    """

    # Initialize lists in which we store the results
    results = list()

    # Loop over all these time points to generate the corresponding indices
    for signal_time in signal_times:

        # Make sure the signal time is an integer (we use it as an index)
        signal_time = int(signal_time)

        # Compute the signal mask for this signal time
        signal_mask = get_signal_mask(
            position=position,
            parang=parang,
            signal_time=signal_time,
            frame_size=frame_size,
            psf_template=psf_template,
            threshold=threshold,
        )

        # Check if the expected signal length is larger than the threshold.
        # In this case, we do not compute the noise and signal masks, but
        # skip this signal time.
        if np.mean(signal_mask) > max_signal_length:
            continue

        # Store the current (signal_time_index, signal_mask, signal_time) tuple
        results.append((signal_mask, signal_time))

    return results


def get_signal_times(n_frames: int, n_signal_times: int) -> np.ndarray:
    """
    Simple function to generate a temporal grid of signal times; mostly
    to ensure consistency everywhere.

    Args:
        n_frames: The total number of frames in the stack.
        n_signal_times: The number of positions on the temporal grid
            that we create.

    Returns:
        A 1D numpy array of shape `(n_signal_times, )` containing the
        temporal grid (i.e., signal times) as integers.
    """

    # Generate `n_signal_times` different possible points in time (distributed
    # uniformly over the observation) at which we planet signal could be
    return np.linspace(0, n_frames - 1, n_signal_times).astype(int)
