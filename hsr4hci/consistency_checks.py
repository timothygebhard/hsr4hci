"""
Utility functions for consistency checks and related tasks.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Tuple, Union

from sklearn.metrics.pairwise import cosine_similarity

from tqdm.auto import tqdm

import h5py
import numpy as np

from hsr4hci.coordinates import get_center, cartesian2polar
from hsr4hci.forward_modeling import add_fake_planet
from hsr4hci.general import find_closest, rotate_position
from hsr4hci.masking import get_positions_from_mask


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_signal_times_from_keys(keys: list) -> np.ndarray:
    return np.array(
        sorted(list(map(int, filter(lambda _: _.isdigit(), keys))))
    )


def get_all_match_fractions(
    dict_or_path: Union[dict, Union[str, Path]],
    roi_mask: np.ndarray,
    hypotheses: np.ndarray,
    parang: np.ndarray,
    psf_template: np.ndarray,
    frame_size: Tuple[int, int],
    n_roi_splits: int = 1,
    roi_split: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Initialize array for the match fractions (mean and median)
    mean_mfs = np.full(frame_size, np.nan)
    median_mfs = np.full(frame_size, np.nan)

    # Define an array in which we keep track of the "affected pixels" (i.e.,
    # the planet traces for every hypothesis), mostly for debugging purposes
    affected_pixels = np.full(frame_size + frame_size, np.nan)

    # Define positions for which to run (= subset of the ROI)
    positions = get_positions_from_mask(roi_mask)[roi_split::n_roi_splits]

    # We can either work with a dict that holds all results...
    if isinstance(dict_or_path, dict):

        # Get signal times
        signal_times = get_signal_times_from_keys(list(dict_or_path.keys()))

        # Loop over (subset of) ROI and compute match fractions
        for position in tqdm(positions, ncols=80):
            mean_mf, median_mf, affected_mask = get_match_fraction(
                position=position,
                hypothesis=hypotheses[position[0], position[1]],
                results=dict_or_path,
                parang=parang,
                psf_template=psf_template,
                signal_times=signal_times,
                frame_size=frame_size,
            )
            mean_mfs[position] = mean_mf
            median_mfs[position] = median_mf
            affected_pixels[position] = affected_mask

    # ...or with the path to an HDF file that holds all results.
    elif isinstance(dict_or_path, (str, Path)):

        # Open the HDF file
        with h5py.File(dict_or_path, 'r') as results:

            # Get signal times
            signal_times = get_signal_times_from_keys(list(results.keys()))

            # Loop over (subset of) ROI and compute match fractions
            for position in tqdm(positions, ncols=80):
                mean_mf, median_mf, affected_mask = get_match_fraction(
                    position=position,
                    hypothesis=hypotheses[position[0], position[1]],
                    results=results,
                    parang=parang,
                    psf_template=psf_template,
                    signal_times=signal_times,
                    frame_size=frame_size,
                )
                mean_mfs[position] = mean_mf
                median_mfs[position] = median_mf
                affected_pixels[position] = affected_mask

    else:
        raise ValueError(
            f'dict_or_path must be dict or Path, is {type(dict_or_path)}!'
        )

    return mean_mfs, median_mfs, affected_pixels


def get_match_fraction(
    position: Tuple[int, int],
    hypothesis: float,
    results: Union[dict, h5py.File],
    parang: np.ndarray,
    psf_template: np.ndarray,
    signal_times: np.ndarray,
    frame_size: Tuple[int, int],
) -> Tuple[float, float, np.ndarray]:
    """
    Compute the match fraction for a single given position.
    """

    n_frames = len(parang)

    # If we do not have a hypothesis for the current position, we can
    # directly return the match fraction as 0
    if np.isnan(hypothesis):
        return 0, 0, np.full(frame_size, False)

    # Compute the expect final position based on the hypothesis that the
    # signal is at `position` at time `signal_time`
    final_position = rotate_position(
        position=position[::-1],  # position is in numpy coordinates
        center=get_center(frame_size),
        angle=float(parang[int(hypothesis)]),
    )

    # Compute the *full* expected signal stack under this hypothesis and
    # normalize it (so that the thresholding below works more reliably!)
    expected_stack = add_fake_planet(
        stack=np.zeros((n_frames,) + frame_size),
        parang=parang,
        psf_template=psf_template,
        polar_position=cartesian2polar(
            position=(final_position[0], final_position[1]),
            frame_size=frame_size,
        ),
        magnitude=0,
        extra_scaling=1,
        dit_stack=1,
        dit_psf_template=1,
        return_planet_positions=False,
        interpolation='bilinear',
    )
    expected_stack = np.asarray(expected_stack / np.max(expected_stack))

    # Find mask of all pixels that are affected by the planet trace, i.e.,
    # all pixels that at some point in time contain planet signal.
    # The threshold value of 0.2 is a bit of a magic number: it serves to
    # pick only those pixels affected by the central peak of the signal,
    # and not the secondary maxima, which are often too low to be picked
    # up by the HSR, and including them would lower the match fraction.
    affected_mask = np.max(expected_stack, axis=0).astype(float) > 0.2

    # Keep track of the matches (similarity scores) for affected positions
    matches = []

    # Convert signal_times to list (to avoid mypy issue with find_closest())
    signal_times_list = list(signal_times)

    # Loop over all affected positions and check how well the residuals
    # match the expected signals
    for (x, y) in get_positions_from_mask(affected_mask):

        # Skip the hypothesis itself: We know that is is a match, because
        # otherwise it would not be our hypothesis
        if (x, y) == position:
            continue

        # Find the time at which this pixel is affected the most
        peak_time = int(np.argmax(expected_stack[:, x, y]))
        _, peak_time = find_closest(signal_times_list, peak_time)

        # Define shortcuts for the time series that we compare
        a = expected_stack[:, x, y]
        b = np.array(results[str(peak_time)]['residuals'][:, x, y])

        # In case we do not have a signal masking residual for the
        # current affected position, we skip it
        if np.isnan(b).any():
            continue

        # Compute the similarity between the expected signal and the
        # actual "best" residual (and make sure it is between 0 and 1)
        similarity = float(
            np.clip(
                cosine_similarity(X=a.reshape(1, -1), Y=b.reshape(1, -1)),
                a_min=0,
                a_max=1,
            )
        )
        matches.append(similarity)

    # TODO: Maybe add a correction factor for pixels where not all pixels
    #       could be checked? (These should perhaps be down-weighted.)

    # Compute mean and median match fraction for current position
    if matches:
        match_fraction__mean = np.nanmean(matches)
        match_fraction__median = np.nanmedian(matches)
    else:
        match_fraction__mean = 0
        match_fraction__median = 0

    return match_fraction__mean, match_fraction__median, affected_mask
