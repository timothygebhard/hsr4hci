"""
Utility functions for consistency checks and related tasks.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Dict, Tuple, Union

from sklearn.metrics.pairwise import cosine_similarity

from tqdm.auto import tqdm

import numpy as np

from hsr4hci.utils.coordinates import get_center, cartesian2polar
from hsr4hci.utils.forward_modeling import add_fake_planet
from hsr4hci.utils.general import find_closest, rotate_position
from hsr4hci.utils.masking import get_positions_from_mask


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_match_fraction(
    hypotheses: np.ndarray,
    results: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]],
    parang: np.ndarray,
    psf_template: np.ndarray,
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
    frame_size = (hypotheses.shape[0], hypotheses.shape[1])
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

        # If we do not have a hypothesis for the current position, we can
        # directly return the match fraction as 0
        if np.isnan(signal_time):
            match_fraction__mean[position] = 0
            match_fraction__median[position] = 0
            continue

        # Compute the expect final position based on the hypothesis that the
        # signal is at `position` at time `signal_time`
        final_position = rotate_position(
            position=position[::-1],  # position is in numpy coordinates
            center=center,
            angle=float(parang[int(signal_time)]),
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
        expected_stack = np.array(expected_stack) / np.max(expected_stack)

        # Find mask of all pixels that are affected by the planet trace, i.e.,
        # all pixels that at some point in time contain planet signal.
        # The threshold value of 0.2 is a bit of a magic number: it serves to
        # pick only those pixels affected by the central peak of the signal,
        # and not the secondary maxima, which are often too low to be picked
        # up by the HSR, and including them would lower the match fraction.
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
            match_fraction__mean[position] = np.nanmean(matches)
            match_fraction__median[position] = np.nanmedian(matches)
        else:
            match_fraction__mean[position] = 0
            match_fraction__median[position] = 0

    return match_fraction__mean, match_fraction__median, affected_pixels
