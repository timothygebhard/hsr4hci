"""
Utilities for finding hypotheses (in the sense: "there is a planet at
position Y at time T").
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Dict, Tuple

from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

from hsr4hci.utils.consistency_checks import get_bump_height
from hsr4hci.utils.forward_modeling import get_time_series_for_position
from hsr4hci.utils.general import fast_corrcoef
from hsr4hci.utils.signal_masking import get_signal_masks


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def find_hypothesis(
    results: Dict[str, np.ndarray],
    position: Tuple[int, int],
    parang: np.ndarray,
    n_signal_times: int,
    frame_size: Tuple[int, int],
    psf_cropped: np.ndarray,
    max_signal_length: float,
    metric_function: str = 'cosine_similarity',
) -> float:
    """
    Take the dictionary containing the full training results and find,
    for a given spatial `position`, the time T at which we suspect there
    to be a planet at the position. If these is no time at which the
    residuals look like the contain a planet, this function returns NaN.

    Args:
        results: A dictionary containing the full training results. In
            particular, we expect that for each signal time T on the
            temporal grid that was used during training, there is a key
            T (as a string) that maps to another dictionary which has a
            key `residuals`, which is a 3D numpy array consisting of the
            residuals that were obtained when trained the signal masking
            models under the hypothesis that there is a planet at T (for
            each spatial position).
        position: A tuple `(x, y)` specifying the position for which we
            want to find the best hypothesis (i.e., the best guess for
            the time at which this pixel contains a planet).
        parang: A 1D numpy array of shape `(n_frames, )` containing the
            parallactic angles for each frame.
        n_signal_times: An integer specifying the number of different
            signal times, that is, the size of the temporal grid that
            was used during training.
        frame_size: A tuple `(width, height)` specifying the size (in
            pixels) of the frames that we are working with.
        psf_cropped: A 2D numpy array containing a cropped version of
            the unsaturated PSF template for the data set.
        max_signal_length: A value in [0.0, 1.0] which describes the
            maximum value of `expected_signal_length / n_frames`, which
            will determine for which pixels we do not want to use the
            "mask out a potential signal region"-approach, because the
            potential signal region is too large to leave us with a
            reasonable amount of training data.
        metric_function: A string specifying the metric function that is
            used to find the best hypothesis. Options are:
                - "bump_height"
                - "correlation_coefficient"
                - "cosine_similarity"

    Returns:
        Either a time T, which is the best guess for the `signal_time`
        of the planet signal in `position`, or `np.nan` if the position
        does not seem to contain any planet signal at any time.
    """

    # Initialize variables in which we store the optimum
    best_signal_time = np.nan
    best_metric = 0.0

    # Loop over the temporal grid to find the best hypothesis
    for signal_mask, signal_time in get_signal_masks(
        position=position,
        parang=parang,
        n_signal_times=n_signal_times,
        frame_size=frame_size,
        psf_cropped=psf_cropped,
        max_signal_length=max_signal_length,
    ):

        # Select residual for this position
        residual = results[str(signal_time)]['residuals'][
            :, position[0], position[1]
        ]

        # Get the value of the metric function at the target position:
        if metric_function == 'bump_height':

            # The bump height metric can be computed directly, because it does
            # not require us to compare the residual with the expected signal
            metric = get_bump_height(
                array=residual,
                signal_time=signal_time,
                signal_mask=signal_mask,
            )

        else:

            # For all other metrics, we first need to compute the expected
            # signal time series to compare it with the residual time series
            expected_time_series = get_time_series_for_position(
                position=position,
                signal_time=signal_time,
                frame_size=frame_size,
                parang=parang,
                psf_template=psf_cropped,
            )

            # Compute the desired target metric
            if metric_function == 'cosine_similarity':
                metric = cosine_similarity(
                    expected_time_series[signal_mask].reshape(1, -1),
                    residual[signal_mask].reshape(1, -1),
                )
            elif metric_function == 'correlation_coefficient':
                metric = fast_corrcoef(
                    expected_time_series[signal_mask].ravel(),
                    residual[signal_mask].ravel(),
                )
            else:
                raise ValueError(
                    'Invalid value for metric_function, must be one of the '
                    'following: "bump_height", "cosine_similarity", '
                    '"correlation_coefficient".'
                )

            # Make sure that `metric` is in [0, 1]. This must NOT be used for
            # the bump_height metric, otherwise we get the wrong maxima!
            metric = np.clip(metric, a_min=0, a_max=1)

        # Store the current metric if it is better than the optimum so far
        if (metric := float(metric)) > best_metric:
            best_metric = metric
            best_signal_time = signal_time

    return float(best_signal_time)
