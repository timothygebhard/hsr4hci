"""
Utility functions for consistency checks and related tasks.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Dict, List, Tuple

from scipy.interpolate import interp1d, RegularGridInterpolator
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

from tqdm.auto import tqdm

import numpy as np

from hsr4hci.utils.forward_modeling import get_time_series_for_position
from hsr4hci.utils.general import fast_corrcoef, find_closest, rotate_position
from hsr4hci.utils.masking import get_positions_from_mask
from hsr4hci.utils.signal_masking import get_signal_mask


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


def get_matches(
    results: Dict[str, np.ndarray],
    hypotheses: np.ndarray,
    parang: np.ndarray,
    psf_cropped: np.ndarray,
    n_test_positions: int,
    metric_function: str = 'cosine_similarity',
    verbose: bool = True,
) -> np.ndarray:
    """
    Construct the match_stack, that is, a 3D numpy array of shape:
        `(n_test_positions, width, height)`
    where each entry describes how well the corresponding test position
    matched the original hypothesis for the respective spatial position.
    All entries are from the interval [0, 1]; the simplest metric based
    on the bump height will only give values from {0, 1}.

    Args:
        results: A dictionary containing the full training results. In
            particular, we expect that for each signal time T on the
            temporal grid that was used during training, there is a key
            T (as a string) that maps to another dictionary which has a
            key `residuals`, which is a 3D numpy array consisting of the
            residuals that were obtained when trained the signal masking
            models under the hypothesis that there is a planet at T (for
            each spatial position).
        hypotheses: A 2D numpy array of shape `(width, height)`, where
            each entry contains the `signal_time` (as an integer that
            can be used as a temporal index) indicating our hypothesis
            for this spatial position (found by the `find_hypothesis()`
            function), that is, basically saying "if there is a planet
            signal at this spatial position, it peaks at this time".
        parang: A numpy array of shape `(n_frames, )` that contains the
            parallactic angles for each frame.
        psf_cropped: A 2D numpy array containing a cropped version of
            the unsaturated PSF template for the data set. This is
            needed to compute the expected signal shape with which we
            compare to determine a match.
        n_test_positions: An integer specifying the desired number of
            test positions along the planet trajectory that is implied
            by the tuple `(position, signal_time)`.
        metric_function: A string defining the metric function that is
            used to determine a match. Must be one of the following:
                - "bump_height"
                - "correlation_coefficient"
                - "cosine_similarity"
            This function is used to check whether at the test positions
            we find a signal that is compatible with the planet path
            hypothesis from the `hypotheses` array.
        verbose: Whether or not to print a progress bar.

    Returns:
        A 3D numpy array of shape `(n_test_positions, width, height)`,
        which for each spatial position contains the result of the
        consistency check at the `n_test_positions` test position (i.e.,
        a number in [0, 1] or {0, 1}, depending on `metric_function`).
    """

    # Define some useful shortcuts
    n_frames = len(parang)
    signal_times = sorted(
        list(map(int, filter(lambda _: _.isdigit(), results.keys())))
    )
    frame_size = hypotheses.shape

    # Define ROI mask: all positions for which we actually have a hypothesis
    roi_mask = np.logical_not(np.isnan(hypotheses))

    # Prepare a grid for the RegularGridInterpolator() below
    t_grid = np.arange(n_frames)
    x_grid = np.arange(frame_size[0])
    y_grid = np.arange(frame_size[1])

    # Initialize array in which we keep track of the test positions which are
    # (not) consistent with the best signal masking-model for each pixel
    matches = np.full((n_test_positions,) + frame_size, np.nan)

    # -------------------------------------------------------------------------
    # Loop over all spatial positions in the ROI and count matches
    # -------------------------------------------------------------------------

    # Get a list of all spatial positions in the ROI; add progress bar
    positions = get_positions_from_mask(roi_mask)
    if verbose:
        positions = tqdm(positions, ncols=80)

    # Loop over all positions in the ROI
    for position in positions:

        # Get the signal time-hypothesis for the current position
        signal_time = hypotheses[position[0], position[1]]

        # Get the test positions for the current hypothesis
        consistency_check_data = get_consistency_check_data(
            position=position,
            signal_time=signal_time,
            parang=parang,
            frame_size=frame_size,
            n_test_positions=n_test_positions,
        )

        # ---------------------------------------------------------------------
        # Loop over test positions and check if they match or not
        # ---------------------------------------------------------------------

        for i, (test_position, test_time) in enumerate(consistency_check_data):

            # Find the closest signal_time for which we actually have trained
            # a model. TODO: Or we could also use interpolation here?
            closest_signal_time = find_closest(signal_times, test_time)

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
                values=results[str(closest_signal_time)]['residuals'],
            )

            # Define the spatio-temporal positions at which we want to retrieve
            # the residual values. By taking only integer values for the first
            # (= temporal) dimension, we are effectively only interpolating the
            # residuals spatially, but not temporally. In other words, for each
            # point in time, we get the residual value by interpolating it from
            # the four closest residual values, using bilinear interpolation.
            residual_positions = np.array(
                [(_,) + test_position for _ in np.arange(n_frames)]
            )

            # Select the interpolated residuals for the current `test_position`
            interpolated_residual = interpolator(residual_positions)

            # The interpolated residual can be all-NaN in cases where the test
            # position is too close to a (spatial) pixel for which we did not
            # train a signal masking model. In this case, we simply ignore this
            # test position (i.e., there will be a NaN in the `matches` array).
            if np.isnan(interpolated_residual).any():
                continue

            # Get the signal mask that matches the current closest_signal_time
            closest_signal_mask = get_signal_mask(
                position=test_position,
                parang=parang,
                signal_time=closest_signal_time,
                frame_size=frame_size,
                psf_cropped=psf_cropped,
            )

            # -----------------------------------------------------------------
            # Check for a match based on the given metric_function
            # -----------------------------------------------------------------

            # The bump_height metric does not require a comparison with the
            # expected planet signal, so we treat it as a special case
            if metric_function == 'bump_height':

                # Compute the bump height at the current test position
                bump_height = get_bump_height(
                    array=interpolated_residual,
                    signal_time=closest_signal_time,
                    signal_mask=closest_signal_mask,
                )

                # This metric is binary: either there is a bump or there is not
                metric = float(bump_height > 0)

            # For all other metric functions, we first need to compute the
            # expected planet signal with which we compare the residual
            else:

                # Compute the expected time series with which we will compare
                expected_time_series = get_time_series_for_position(
                    position=position,
                    signal_time=closest_signal_time,
                    frame_size=frame_size,
                    parang=parang,
                    psf_template=psf_cropped,
                )

                # Define shortcuts for the time series that we compare
                x = expected_time_series[closest_signal_mask].reshape(1, -1)
                y = interpolated_residual[closest_signal_mask].reshape(1, -1)

                # Compute the desired target metric by comparing the residual
                # with the expected planet signal
                if metric_function == 'cosine_similarity':
                    metric = cosine_similarity(x, y)
                elif metric_function == 'correlation_coefficient':
                    metric = fast_corrcoef(x.ravel(), y.ravel())
                else:
                    raise ValueError(
                        'Invalid value for metric_function, must be one of '
                        'the following: "bump_height", "cosine_similarity", '
                        '"correlation_coefficient".'
                    )

            # Make sure that `metric` is in [0, 1]. Here, we can also clip the
            # value of `metric` in case we use the bump height as the metric
            # because we only check if there is a bump, and do not look for the
            # "best" bump (unlike in the find_hypothesis() function).
            metric = float(np.clip(metric, a_min=0, a_max=1))

            # Store whether or not there is a bump at the test position
            matches[i, position[0], position[1]] = metric

    return matches
