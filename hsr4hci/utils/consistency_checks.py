"""
Utility functions for consistency checks and related tasks.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Dict, Tuple

from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

from tqdm.auto import tqdm

import numpy as np

from hsr4hci.utils.coordinates import get_center, cartesian2polar
from hsr4hci.utils.forward_modeling import add_fake_planet
from hsr4hci.utils.general import fast_corrcoef, find_closest, rotate_position
from hsr4hci.utils.masking import get_positions_from_mask


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


def get_match_fraction(
    hypotheses: np.ndarray,
    results: Dict[str, np.ndarray],
    parang: np.ndarray,
    psf_template: np.ndarray,
    metric_function: str = 'cosine_similarity',
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loop over the `hypotheses` and compute, for each pixel, the match
    fraction, that is, how consistent this hypothesis is with the rest
    of the `results`.

    The idea is the following: For each pixel Y, we have a hypothesis
    of the form (Y, T). For this, we use the `add_fake_planet()` method
    to compute the *full* expected signal stack under this hypothesis.
    We then determine which spatial pixels should be affected by the
    planet at which times (i.e., we compute the "sausage"-shaped trace
    of the planet signal). For each of these affected pixels, we then
    compute the similarity (= how well it matches) with the respective
    signal masking residual in `results` (using the `metric_function`).
    Finally, we compute the match fraction (mean and median) for Y as
    the average of the similarity scores ("matches") of all affected
    pixels.

    Args:
        hypotheses: A 2D numpy array with shape `(x_size, y_size)` where
            each pixel contains our hypothesis for when this pixel might
            contain a planet signal (in the form of an integer denoting
             a temporal index).
        results: A dictionary containing the full training results. In
            particular, we expect that for each signal time T on the
            temporal grid that was used during training, there is a key
            T (as a string) that maps to another dictionary which has a
            key `residuals`, which is a 3D numpy array consisting of the
            residuals that were obtained when trained the signal masking
            models under the hypothesis that there is a planet at T (for
            each spatial position).
        parang: A 1D numpy array of shape `(n_frames, )` containing the
            parallactic angle.
        psf_template: A 2D numpy array containing the PSF template.
        metric_function: A string containing the metric that is used to
            measure the similarity between an expected signal and the
            observed residual. Must be one of the following:
                "cosine_similarity" (default), "bump_height",
                "correlation_coefficient"
        verbose: Whether or not to show a progress bar.

    Returns:
        match_fraction__mean: A 2D numpy array containing the match
            fraction for every pixel computed as the mean.
        match_fraction__median: A 2D numpy array containing the match
            fraction for every pixel computed as the median.
        affected_pixels A 4D numpy array of shape `(x_size, y_size,
            x_size, y_size)` which, for each pixel, contains a 2D
            numpy array indicating the pixels that would, under the
            hypothesis for this pixel, be affected by the signal.
    """

    # Define some useful shortcuts
    n_frames = len(parang)
    signal_times = sorted(
        list(map(int, filter(lambda _: _.isdigit(), results.keys())))
    )
    frame_size = hypotheses.shape
    center = get_center(frame_size)

    # Define ROI mask: all positions for which we actually have a hypothesis
    roi_mask = np.logical_not(np.isnan(hypotheses))

    # Make sure that the PSF template is normalized to maximum of 1. This is
    # necessary because we later determine the affected pixels by thresholding
    # the expected signal stack.
    psf_template -= np.nanmin(psf_template)
    psf_template /= np.nanmax(psf_template)

    # -------------------------------------------------------------------------
    # Loop over all spatial positions in the ROI and compute match fraction
    # -------------------------------------------------------------------------

    # Define result arrays: mean and median match fraction
    match_fraction__mean = np.full(frame_size, np.nan)
    match_fraction__median = np.full(frame_size, np.nan)

    # Define an array in which we keep track of the "affected pixels" (i.e.,
    # the planet traces for every hypothesis) for debugging purposes
    affected_pixels = np.full(frame_size + frame_size, np.nan)

    # Get a list of all spatial positions in the ROI; add progress bar
    positions = get_positions_from_mask(roi_mask)
    if verbose:
        positions = tqdm(positions, ncols=80)

    # Loop over all positions in the ROI
    for position in positions:

        # Get the signal time-hypothesis for the current position
        signal_time = hypotheses[position[0], position[1]]

        # Compute the expect final position based on the hypothesis that the
        # signal is at `position` at time `signal_time`
        final_position = rotate_position(
            position=position[::-1],  # position is in numpy coordinates
            center=center,
            angle=float(parang[int(signal_time)]),
        )

        # Compute the *full* expected signal stack under this hypothesis
        expected_stack = add_fake_planet(
            stack=np.zeros((n_frames, ) + frame_size),
            parang=parang,
            psf_template=psf_template,
            polar_position=cartesian2polar(
                position=final_position, frame_size=frame_size,
            ),
            magnitude=0,
            extra_scaling=1,
            dit_stack=1,
            dit_psf_template=1,
            return_planet_positions=False,
            interpolation='bilinear',
        )
        expected_stack = np.array(expected_stack)

        # Find mask of all pixels that are affected by the planet trace, i.e.,
        # all pixels that at some point in time contain planet signal.
        # The threshold value of 0.2 is a bit of a magic number: it serves to
        # pick only those pixels affected by the central peak of the signal,
        # and not the secondary maxima.
        affected_mask = np.max(expected_stack, axis=0).astype(float) > 0.2

        # Keep track of the matches (similarity scores) for affected positions
        matches = []

        # Loop over all affected positions and check how well the residuals
        # match the expected signals
        for (x, y) in get_positions_from_mask(affected_mask):

            # Skip the hypothesis itself: We know that is is a match, because
            # otherwise it would not be our hypothesis
            if (x, y) == position:
                continue

            # Find the time at which this pixel is affected the most
            peak_time = int(np.argmax(expected_stack[:, x, y]))
            _, peak_time = find_closest(signal_times, peak_time)
            affected_pixels[position[0], position[1], x, y] = peak_time

            # Define shortcuts for the time series that we compare
            a = expected_stack[:, x, y]
            b = results[str(peak_time)]['residuals'][:, x, y]

            # In case we do not have a signal masking residual for the
            # current affected position, we skip it
            if np.isnan(b).any():
                continue

            # Compute the desired target metric by comparing the residual
            # with the expected planet signal
            if metric_function == 'bump_height':
                signal_mask = b > 0.2
                metric = get_bump_height(b, signal_mask, peak_time)
            elif metric_function == 'cosine_similarity':
                metric = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
            elif metric_function == 'correlation_coefficient':
                metric = fast_corrcoef(a, b)
            else:
                raise ValueError(
                    'Invalid value for metric_function, must be one of '
                    'the following: "bump_height", "cosine_similarity", '
                    '"correlation_coefficient".'
                )

            # Make sure that `metric` is in [0, 1]. Here, we can also clip the
            # value of `metric` in case we use the bump height as the metric
            # because we only check if there is a bump, and do not look for
            # the "best" bump (unlike in the find_hypothesis() function).
            metric = float(np.clip(metric, a_min=0, a_max=1))
            matches.append(metric)

        # Compute "correction factor" to down-weigh those pixels where not
        # all affected pixels could be checked for a match
        # TODO: Discuss if this factor makes sense / is fair
        factor = np.sqrt(len(matches) / (np.nansum(affected_mask) - 1))

        # Compute mean and median match fraction for current position
        match_fraction__mean[position] = factor
        match_fraction__median[position] = np.nanmedian(matches) * factor

    return match_fraction__mean, match_fraction__median, affected_pixels
