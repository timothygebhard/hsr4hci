"""
Utility functions for consistency checks and related tasks.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import List, Tuple

from scipy.interpolate import interp1d, RegularGridInterpolator
from sklearn.linear_model import LinearRegression

from tqdm.auto import tqdm

import numpy as np

from hsr4hci.utils.general import rotate_position
from hsr4hci.utils.masking import get_positions_from_mask
from hsr4hci.utils.signal_masking import get_signal_length


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_bump_height(
    array: np.ndarray,
    signal_mask: np.ndarray,
    signal_time: int,
) -> float:
    """
    Check if a given `array` (typically residuals) has a positive bump
    in the region that is indicated by the given `signal_mask`, and
    return the `bump_height` (which defaults to 0 if there is no bump).

    To check if the selected region has a bump-like shape, we use a very
    simple heuristic: We split the search region into two parts, based
    on the given `signal_time`, and fit both parts with a linear model.
    If the first regression returns a positive slope, and the second
    regression returns a negative slope, we compute the bump height as
    the median of the search region; otherwise, we return 0. We also
    return 0 if the median of the search region is actually negative.

    Args:
        array: A 1D numpy array in which we search for a bump.
        signal_mask: A 1D numpy array indicating the search region.
        signal_time: The index specifying the "exact" location
            where the peak of the bump should be located.

    Returns:
        The height of the bump (computed as the median of the part of
        `array` selected by the `signal_mask`, or zero if the median is
        negative), in case there is a positive bump; otherwise zero.
    """

    # Get the start and end position of the signal_mask
    all_idx = np.arange(len(array))
    signal_start, signal_end = all_idx[signal_mask][np.array([0, -1])]

    # Prepare predictors and targets for the two linear fits
    predictors_1 = np.arange(signal_start, signal_time).reshape(-1, 1)
    predictors_2 = np.arange(signal_time, signal_end).reshape(-1, 1)
    targets_1 = array[signal_start:signal_time]
    targets_2 = array[signal_time:signal_end]

    # Fit regions with linear models and get slopes
    if len(predictors_1) > 2:
        model_1 = LinearRegression().fit(predictors_1, targets_1)
        slope_1 = model_1.coef_[0]
    else:
        slope_1 = 1
    if len(predictors_2) > 2:
        model_2 = LinearRegression().fit(predictors_2, targets_2)
        slope_2 = model_2.coef_[0]
    else:
        slope_2 = -1

    # Define criteria for "is there a positive bump at the given signal time?"
    criterion_1 = bool(slope_1 > 0 > slope_2)
    criterion_2 = bool(
        np.nanmedian(array[signal_mask]) > np.nanmedian(array[~signal_mask])
    )

    # If there is a bump, we return its height; otherwise we return 0
    if criterion_1 and criterion_2:
        bump_height = float(np.nanmedian(array[signal_mask]))
        return max((0, bump_height))
    return 0


def get_consistency_check_data(
    position: Tuple[float, float],
    signal_time: float,
    parang: np.ndarray,
    frame_size: Tuple[int, int],
    n_test_positions: int = 5,
) -> List[Tuple[Tuple[float, float], float]]:
    """
    Given a (spatial) `position` and a (temporal) `signal_time`, infer
    the planet path that is implied by these values and return a list of
    test positions that are on that arc, together with the respective
    expected temporal signal position at these positions.

    Args:
        position: A tuple `(x, y)` indicating the position at which we
            believe the planet is at the given `signal_time`.
        signal_time: The time (in the form of a temporal index for the
            `parang` array) at which we think there is a planet signal
            at the given `position`.
        parang: A numpy array of shape `(n_frames, )` that contains the
            parallactic angles for each frame.
        frame_size: A tuple of integers, `(width, height)`, indicating
            the (spatial) size of the frames that we are working with.
        n_test_positions: An integer specifying the desired number of
            test positions along the planet trajectory that is implied
            by the tuple `(position, signal_time)`.

    Returns:
        A list of `n_test_positions` 2-tuples, where each tuple contains
        one spatio-temporal test position: `(test_position, test_time)`.
    """

    # Define useful shortcuts
    n_frames = len(parang)
    center = (frame_size[0] / 2, frame_size[1] / 2)

    # Create a (linear) interpolator for parang such that we can evaluate
    # the parallactic angle at arbitrary times
    interpolate_parang = interp1d(np.arange(n_frames), parang)

    # Assuming that the peak of the signal is at pixel `position` at the time
    # t = `signal_time`, use our knowledge about the movement of the planet to
    # compute the (spatial) position of the planet at point t = 0.
    starting_position = rotate_position(
        position=position,
        center=center,
        angle=-float(interpolate_parang(signal_time) - parang[0]),
    )

    # Create `n_test_times` (uniformly distributed in time) points at which we
    # check if the find a planet signal consistent with the above hypothesis
    test_times = np.linspace(0, n_frames - 1, n_test_positions)

    # Compute the positions that correspond to these times
    test_positions = np.array(
        rotate_position(
            position=starting_position,
            center=center,
            angle=interpolate_parang(test_times) - parang[0],
        )
    ).transpose()

    # Combine the test_positions and test_times into a list of tuples
    return [((p[0], p[1]), t) for p, t in zip(test_positions, test_times)]

        # Collect and store result tuple
        result = (test_position, test_time, test_mask)
        results.append(result)

    return results


def get_match_fraction(
    results: dict,
    parang: np.ndarray,
    psf_diameter: float,
    roi_mask: np.ndarray,
    n_test_positions: int,
) -> np.ndarray:
    """
    Run consistency tests to compute the fraction of matches for each
    position. This is the basis for choosing for which pixels we use the
    baseline model and for which we use the best signal masking model.

    Args:
        results:
        parang:
        psf_diameter:
        roi_mask:
        n_test_positions:

    Returns:

    """

    # Define some useful shortcuts
    signal_times = results['signal_times']
    n_frames = len(parang)
    frame_size = roi_mask.shape

    # Prepare a grid for the RegularGridInterpolator() below
    t_grid = np.arange(n_frames)
    x_grid = np.arange(frame_size[0])
    y_grid = np.arange(frame_size[1])

    # Initialize array in which we keep track of the fraction of test positions
    # which are consistent with the best signal masking-model for each pixel
    match_fraction = np.full(frame_size, np.nan)

    # Loop over all positions in the ROI
    for position in tqdm(get_positions_from_mask(roi_mask), ncols=80):

        # Check if the best signal time for this pixel is NaN, which implies
        # that for this pixel there exists no best signal-masking model. In
        # this case, we do not have to run consistency checks and can set the
        # match fraction directly to 0.
        signal_time = results['best']['signal_time'][position[0], position[1]]
        if np.isnan(signal_time):
            match_fraction[position[0], position[1]] = 0
            continue

        # Otherwise, we can convert the best signal time to an integer
        signal_time = int(signal_time)

        # Get the data for the consistency check, that is, a list of spatial
        # positions and temporal indices (plus masks) at which we also expect
        # to see a planet if the hypothesis about the planet path that is
        # implied by the above best signal masking-model is correct.
        consistency_check_data = get_consistency_check_data(
            position=position,
            signal_time=signal_time,
            parang=parang,
            frame_size=frame_size,
            psf_diameter=psf_diameter,
            n_test_positions=n_test_positions,
        )

        # Initialize a list to keep track of the matching test position
        matches: List[int] = list()

        # Loop over all test positions and perform the consistency check
        for (test_position, test_time, test_mask) in consistency_check_data:

            # We have only trained signal masking models for a finite set of
            # possible signal times. Out of those, we now find the one that is
            # the closest to the current `test_time`, because we will be using
            # the residuals obtained from this model for the consistency check.
            closest_signal_time = np.abs(
                signal_times.ravel() - test_time
            ).argmin()

            # Set up an interpolator for the residuals of the model that was
            # trained assuming the signal was at `closest_signal_time`.
            # Rationale: Most `test_positions` will not exactly match one of
            # the spatial positions for which we have trained a model and
            # computed residuals. If we simply round the `test_position` to
            # the closest integer position, we will likely get duplicates for
            # higher values of `n_test_positions`, which might introduce a bias
            # to the match fraction. Setting up this interpolator circumvents
            # this because it allows us to get the value of the the residuals
            # at *arbitrary* spatio-temporal positions, thus removing the need
            # to round the `test_position` to the closest integer position.
            interpolator = RegularGridInterpolator(
                points=(t_grid, x_grid, y_grid),
                values=results[str(closest_signal_time)]['residuals']
            )

            # Define the spatio-temporal positions at which we want to retrieve
            # the residual values. By taking only integer values for the first
            # (= temporal) dimension, we are effectively only interpolating the
            # residuals spatially, but not temporally. In other words, for each
            # point in time, we get the residual value by interpolating it from
            # the four closest residual values, using bilinear interpolation.
            residual_positions = np.array(
                [(_, ) + test_position for _ in np.arange(n_frames)]
            )

            # Select the interpolated residuals for the current `test_position`
            residuals = interpolator(residual_positions)

            # Check if the residuals contain NaN, which would indicate that
            # the test position is outside of the ROI, or was so close to the
            # star that the `max_signal_length` threshold has caused us not to
            # learn this model.
            # In this case, we default to "failed the consistency test" to
            # avoid artifacts in the final match_fraction array.
            if np.isnan(residuals).any():
                matches.append(0)
                continue

            # Split the residual into the part that should contain a planet
            # signal and a part that should only contain residual noise
            signal_residual = residuals[test_mask]
            noise_residual = residuals[~test_mask]

            # This is the centerpiece of our consistency check: We define our
            # criteria for counting a match:
            # 1. Do the residuals have a bump at the expected position?
            # 2. Is the mean of the (expected) signal part of the residual
            #    greater than than of the (expected) noise part?
            criterion_1 = has_bump(residuals, test_mask, test_time)
            criterion_2 = np.mean(signal_residual) > np.mean(noise_residual)

            # Only if all criteria are True do we count the current test
            # position as a match for the consistency check
            if criterion_1 and criterion_2:
                matches.append(1)
            else:
                matches.append(0)

        # Finally, compute the match fraction for the current position
        if matches:
            match_fraction[position[0], position[1]] = np.mean(matches)
        else:
            match_fraction[position[0], position[1]] = 0

    return match_fraction
