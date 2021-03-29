"""
Utilities for finding hypotheses of the form (Y, T), that is, pixel Y
seems to contain a planet signal at time T.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Dict, Tuple, Union

from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

import numpy as np

from hsr4hci.forward_modeling import get_time_series_for_position
from hsr4hci.masking import get_positions_from_mask
from hsr4hci.signal_masking import get_signal_times


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_all_hypotheses(
    roi_mask: np.ndarray,
    results: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]],
    parang: np.ndarray,
    n_signal_times: int,
    frame_size: Tuple[int, int],
    psf_template: np.ndarray,
    minimum_similarity: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This is a convenience function which wraps the loop over the ROI
    to call `find_hypothesis()` for every spatial pixel. See there for
    a full documentation of all parameters.
    """

    # Initialize array for hypotheses and similarities
    hypotheses = np.full(frame_size, np.nan)
    similarities = np.full(frame_size, np.nan)

    # Loop over all spatial positions and find the respective hypothesis
    for position in tqdm(get_positions_from_mask(roi_mask), ncols=80):

        signal_time, similarity = find_hypothesis(
            results=results,
            position=position,
            parang=parang,
            n_signal_times=n_signal_times,
            frame_size=frame_size,
            psf_template=psf_template,
            minimum_similarity=minimum_similarity,
        )
        hypotheses[position[0], position[1]] = signal_time
        similarities[position[0], position[1]] = similarity

    return hypotheses, similarities


def find_hypothesis(
    results: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]],
    position: Tuple[int, int],
    parang: np.ndarray,
    n_signal_times: int,
    frame_size: Tuple[int, int],
    psf_template: np.ndarray,
    minimum_similarity: float = 0.0,
) -> Tuple[float, float]:
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
        psf_template: A 2D numpy array containing the unsaturated PSF
            template for the data set.
        minimum_similarity: Minimum cosine similarity between the
            expected and the observed signal for the signal to be
            counted as a hypothesis.

    Returns:
        A tuple `(signal_time, similarity)`, where `signal_time` is
        the best guess for the `signal_time` of the planet signal in
        `position`, and `score` is the corresponding cosine similarity
        between the expected and the observed signal.
        If the `position` does not seem to contain a planet signal at
        any time, both of these values are NaN.
    """

    # Initialize variables in which we store the optimum
    best_signal_time = np.nan
    best_similarity = 0.0

    # Loop over the temporal grid to find the best hypothesis
    for signal_time in get_signal_times(
        n_frames=len(parang), n_signal_times=n_signal_times
    ):

        # Select residual for this position
        signal_time_residuals = results[str(signal_time)]['residuals']
        residual = signal_time_residuals[:, position[0], position[1]]

        # If the residual is NaN, we can't compute the metric function
        if np.isnan(residual).any():
            continue

        # Compute the expected signal time series
        expected_time_series = get_time_series_for_position(
            position=position,
            signal_time=signal_time,
            frame_size=frame_size,
            parang=parang,
            psf_template=psf_template,
        )

        # Compute the cosine similarity, which serves as a metric for how
        # well the expected time series and the residual match
        similarity = cosine_similarity(
            expected_time_series.reshape(1, -1),
            residual.reshape(1, -1),
        )

        # Clip the similarity to [0, 1] (a signal time that leads to a
        # negative cosine similarity should never become a hypothesis)
        similarity = float(np.clip(similarity, a_min=0, a_max=1))

        # Store the current similarity if it is better than the optimum so far
        if similarity > best_similarity:
            best_similarity = similarity
            best_signal_time = signal_time

    # Return type must be float, because it can also be NaN
    if best_similarity > minimum_similarity:
        return float(best_signal_time), best_similarity
    return np.nan, np.nan
