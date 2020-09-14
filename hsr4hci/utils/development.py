"""
Experimental utility functions that are still under heavy development,
and for which it is not fully clear whether or not they will actually
be useful in the end.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from cmath import polar
from typing import List, Optional, Tuple

from sklearn.linear_model import LinearRegression

import numpy as np

from hsr4hci.utils.fitting import moffat_2d, fit_2d_function
from hsr4hci.utils.general import crop_center, rotate_position


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def smooth(array: np.ndarray, window_size: int = 10) -> np.ndarray:
    """
    Smooth by rolling average: Convolve the given `array` with a
    rectangle function of the given `window_size`

    Args:
        array: A 1D numpy array.
        window_size: A positive integer, specifying the width of the
            rectangle with which `array` is convolved.

    Returns:
        A smoothed version of `array`.
    """

    return np.convolve(array, np.ones(window_size) / window_size, mode='same')


def get_effective_pixel_width(
    position: Tuple[int, int], center: Tuple[float, float],
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
    position: Tuple[int, int],
    signal_position: int,
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

    Taking the `frame_idx` into account is necessary, because the
    temporal derivative of the parallactic angle is non-constant over
    the course of an observation, but can change up to around 50% for
    some data sets.

    In theory, a more exact estimate for this number can be achieved
    using proper forward modeling, but that takes about O(10^4) times
    longer, and still involves some arbitrary choices, meaning that
    there is no real *true* value anyway.

    Args:
        position:
        signal_position:
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
    value_1 = parang[signal_position] - np.rad2deg(gamma) / 2
    value_2 = parang[signal_position] + np.rad2deg(gamma) / 2
    if ascending:
        position_1 = np.searchsorted(parang, value_1, side='left')
        position_2 = np.searchsorted(parang, value_2, side='right')
    else:
        position_2 = np.searchsorted(-parang, -value_1, side='left')
        position_1 = np.searchsorted(-parang, -value_2, side='right')

    # Compute the length before and after the peak (because the signal will, in
    # general, not be symmetric around the peak)
    length_1 = int(1.2 * (signal_position - position_1))
    length_2 = int(1.2 * (position_2 - signal_position))

    return length_1, length_2


def get_noise_signal_idx(
    position: Tuple[int, int],
    parang: np.ndarray,
    n_signal_positions: int,
    frame_size: Tuple[int, int],
    psf_diameter: float,
    signal_length_threshold: float = 0.7,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Loop over possible planet positions (in a time series), and for each
    presumed position, return the corresponding train and apply indices.
    Train indices are those that, under the assumption regarding the
    signal position, do not contain planet signal, while the apply
    indices are their complement.

    The results are returned as binary arrays that can, for example, be
    used to select frames from the stack.

    Args:
        position: An integer tuple `(x, y)` specifying the position of
            the pixel for which we are computing the indices.
        parang: A numpy array of shape `(n_frames, )` containing the
            parallactic angles.
        n_signal_positions: The number of different possible planet
            positions for which to return indices.
        frame_size: A tuple `(width, height)` specifying the spatial
            size of the stack.
        psf_diameter: The diameter of the PSF template (in pixels).
        signal_length_threshold: A value in [0.0, 1.0] which describes
            the maximum value of `expected_signal_length / n_frames`,
            which wil determine for which pixels we do not want to use
            the "exclude a potential signal region"-approach, because
            the potential signal region is too large to leave us with a
            reasonable amount of training data.

    Returns:
        This function returns a list of up to `n_position` 3-tuples
        of the following form:
            `(noise_idx, signal_idx, signal_position)`.
        The first two elements are 1D binary numpy arrays of length
        `n_frames`, wheres the last element is an integer giving the
        position of the peak of the planet signal.
    """

    # Define shortcuts
    n_frames = len(parang)
    center = (frame_size[0] / 2, frame_size[1] / 2)

    # Initialize lists in which we store the results
    results = list()

    # Generate `n_positions` different possible signal positions (i.e.,
    # temporal indices specifying the center of the planet signal) that are
    # uniformly distributed over the entire time series
    signal_positions = np.linspace(0, n_frames, n_signal_positions + 2)[1:-1]

    # Loop over all positions to generate the corresponding indices
    for signal_position in signal_positions:

        # Make sure the signal position is an integer
        signal_position = int(signal_position)

        # Compute the expected signal length at this position and index
        length_1, length_2 = get_signal_length(
            position=position,
            signal_position=signal_position,
            center=center,
            parang=parang,
            psf_diameter=psf_diameter,
        )

        # Check if the expected signal length is larger than the threshold.
        # In this case, we do not compute the noise and signal indices, but
        # skip this signal position entirely.
        if (length_1 + length_2) / n_frames > signal_length_threshold:
            return []

        # Initialize apply_idx as all False
        signal_idx = np.zeros(n_frames).astype(bool)

        # Now add a block of 1s (that matches the expected signal length) to
        # the apply_idx centered on the current signal position
        position_1 = max(0, signal_position - length_1)
        position_2 = min(n_frames, signal_position + length_2)
        signal_idx[position_1:signal_position] = True
        signal_idx[signal_position:position_2] = True

        # Compute train_idx as the complement of the apply_idx
        noise_idx = np.logical_not(signal_idx)

        # Store the current (noise_idx, signal_idx, signal_position) tuple
        results.append((noise_idx, signal_idx, signal_position))

    return results


def get_psf_diameter(
    psf_template: np.ndarray,
    pixscale: Optional[float] = None,
    lambda_over_d: Optional[float] = None,
) -> float:
    """
    Fit a 2D Moffat function to the given PSF template to estimate
    the diameter of the central "blob" in pixels.

    The diameter is computed at the arithmetic mean of the FWHM in
    x and y direction, as returned by the fit.

    Args:
        psf_template: A 2D numpy array containing the raw, unsaturated
            PSF template.
        pixscale:
        lambda_over_d:

    Returns:
        The diameter of the PSF template in pixels.
    """

    # Case 1: We have been provided a suitable PSF template and can determine
    # the size by fitting the PSF with a Moffat function
    if psf_template.shape[0] >= 33 and psf_template.shape[1] >= 33:

        # Crop PSF template: too large templates (which are mostly zeros) can
        # cause problems when fitting them with a 2D Moffat function
        psf_template = crop_center(psf_template, (33, 33))

        # Define shortcuts
        psf_center_x = float(psf_template.shape[0] / 2)
        psf_center_y = float(psf_template.shape[1] / 2)

        # Define initial guess for parameters
        p0 = (psf_center_x, psf_center_y, 1, 1, 1, 0, 0, 1)

        # Fit the PSF template with a 2D Moffat function to get the FWHMs
        params = fit_2d_function(frame=psf_template, function=moffat_2d, p0=p0)

        # Compute the PSF diameter as the mean of the two FWHM values
        fwhm_x, fwhm_y = params[2:4]
        psf_diameter = float(0.5 * (fwhm_x + fwhm_y))

    # Case 2: We do not have PSF template, but the PIXSCALE and LAMBDA_OVER_D
    elif (pixscale is not None) and (lambda_over_d is not None):

        # In this case, we can approximately compute the expected PSF size.
        # The 1.144 is a magic number to get closer to the empirical estimate
        # from data sets where a PSF template is available.
        psf_diameter = lambda_over_d / pixscale * 1.144

    # Case 3: In all other scenarios, we raise an error
    else:
        raise RuntimeError('Could not determine PSF diameter')

    return psf_diameter


def has_bump(
    array: np.ndarray, signal_idx: np.ndarray, signal_position: int,
) -> bool:
    """
    Check if a given `array` (typically residuals) has a positive bump
    in the region that is indicated by the given `idx`.

    Currently, the used heuristic is extremely simple:
    We split the search region into two parts, based on the given
    `signal_position`, and fit both parts with a linear model.
    If the first regression returns a positive slope, and the second
    regression returns a negative slope, the function returns True.

    Args:
        array: A 1D numpy array in which we search for a bump.
        signal_idx: A 1D numpy array indicating the search region.
        signal_position: The index specifying the "exact" location
            where the peak of the bump should be located.

    Returns:
        Whether or not the given `array` contains a positive bump in
        the given search region.
    """

    # Get the start and end position of the signal_idx
    all_idx = np.arange(len(array))
    signal_start, signal_end = all_idx[signal_idx][np.array([0, -1])]

    # Prepare predictors and targets for the two linear fits
    predictors_1 = np.arange(signal_start, signal_position).reshape(-1, 1)
    predictors_2 = np.arange(signal_position, signal_end).reshape(-1, 1)
    targets_1 = array[signal_start:signal_position]
    targets_2 = array[signal_position:signal_end]

    # Fit the two models to the two parts of the search region and get the
    # slope of the model
    if len(predictors_1) > 0:
        model_1 = LinearRegression().fit(predictors_1, targets_1)
        slope_1 = model_1.coef_[0]
    else:
        slope_1 = 1
    if len(predictors_2) > 0:
        model_2 = LinearRegression().fit(predictors_2, targets_2)
        slope_2 = model_2.coef_[0]
    else:
        slope_2 = -1

    return bool(slope_1 > 0 > slope_2)


def get_consistency_check_data(
    position: Tuple[int, int],
    signal_position: np.ndarray,
    parang: np.ndarray,
    frame_size: Tuple[int, int],
    psf_diameter: float,
    n_test_positions: int = 5,
) -> List[Tuple[Tuple[int, int], int, np.ndarray]]:
    """
    Given a (spatial) `position` and a (temporal) `signal_position`,
    construct the planet path that is implied by these values and
    return `n_test_positions` new positions on that arc with the
    respective expected temporal signal position at these positions.

    Args:
        position:
        signal_position:
        parang:
        frame_size:
        psf_diameter:
        n_test_positions:

    Returns:

    """

    # Define useful shortcuts
    n_frames = len(parang)
    center = (frame_size[0] / 2, frame_size[1] / 2)

    # Assuming that the peak of the signal is at pixel `position` at the time
    # t=`signal_position`, use our knowledge about the movement of the planet
    # to compute the (spatial) position of the planet at point t=0.
    starting_position = rotate_position(
        position=position,
        center=center,
        angle=-float(parang[signal_position] - parang[0])
    )

    # Create `n_test_times` (uniformly distributed) points in time at which
    # we check if the find a planet signal consistent with the above hypothesis
    test_times = np.linspace(0, n_frames - 1, n_test_positions)
    test_times = test_times.astype(int)

    # Loop over all test positions and get the expected position (both the peak
    # position and temporal region that is covered by the signal) of the signal
    results = list()
    for test_time in test_times:

        # Find the expected (spatial) position
        test_position = rotate_position(
            position=starting_position,
            center=center,
            angle=float(parang[test_time] - parang[0])
        )

        # Round to the closest pixel position
        test_position = (int(test_position[0]), int(test_position[1]))

        # Get the expected signal length at this position
        length_1, length_2 = get_signal_length(
            position=test_position,
            signal_position=test_time,
            center=center,
            parang=parang,
            psf_diameter=psf_diameter,
        )

        # Initialize expected_mask as all False
        test_mask = np.zeros(n_frames).astype(bool)

        # Now add a block of 1s (that matches the expected signal length) to
        # the apply_idx centered on the current signal position
        time_1 = max(0, test_time - length_1)
        time_2 = min(n_frames, test_time + length_2)
        test_mask[time_1:test_time] = True
        test_mask[test_time:time_2] = True

        # Collect and store result tuple
        result = (test_position, test_time, test_mask)
        results.append(result)

    return results
