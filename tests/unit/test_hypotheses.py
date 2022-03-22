"""
Tests for hypotheses.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Dict

from astropy.modeling import models

import numpy as np

from hsr4hci.forward_modeling import get_time_series_for_position
from hsr4hci.hypotheses import (
    get_all_hypotheses,
    get_hypothesis_for_position,
)
from hsr4hci.masking import get_circle_mask
from hsr4hci.training import get_signal_times


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__get_hypothesis_for_position() -> None:
    """
    Test `hsr4hci.hypotheses.get_hypothesis_for_position`.
    """

    # Fix random seed
    np.random.seed(42)

    # Create fake PSF template
    x, y = np.meshgrid(np.arange(33), np.arange(33))
    model = models.Gaussian2D(x_mean=17, y_mean=17)
    psf_template = model(x, y)

    # Create test data
    n_frames, x_size, y_size = 50, 10, 10
    parang = np.linspace(0, 90, n_frames)
    n_signal_times = 5
    signal_times = get_signal_times(n_frames, n_signal_times)
    residuals: Dict[str, np.ndarray] = {}
    for i, signal_time in enumerate(signal_times):

        # For all cases
        tmp_residuals = np.random.normal(0, 1, (n_frames, x_size, y_size))
        tmp_residuals = tmp_residuals.astype(np.float32)

        # For case 1
        if i == 2:
            tmp_residuals[:, 3, 4] = get_time_series_for_position(
                position=(3, 4),
                signal_time=signal_time,
                frame_size=(x_size, y_size),
                parang=parang,
                psf_template=psf_template,
            )

        # For case 2
        tmp_residuals[3, 2, 7] = np.nan

        # For case 4
        tmp_residuals[:, 8, 2] = -1 * get_time_series_for_position(
            position=(8, 2),
            signal_time=signal_time,
            frame_size=(x_size, y_size),
            parang=parang,
            psf_template=psf_template,
        )

        residuals[str(signal_time)] = tmp_residuals

    # Case 1
    signal_time, similarity = get_hypothesis_for_position(
        residuals=residuals,
        position=(3, 4),
        parang=parang,
        n_signal_times=n_signal_times,
        frame_size=(x_size, y_size),
        psf_template=psf_template,
        minimum_similarity=0,
    )
    assert signal_time == 24
    assert np.isclose(similarity, 1)

    # Case 2
    signal_time, similarity = get_hypothesis_for_position(
        residuals=residuals,
        position=(2, 7),
        parang=parang,
        n_signal_times=n_signal_times,
        frame_size=(x_size, y_size),
        psf_template=psf_template,
        minimum_similarity=0,
    )
    assert np.isnan(signal_time)
    assert np.isnan(similarity)

    # Case 3
    signal_time, similarity = get_hypothesis_for_position(
        residuals=residuals,
        position=(4, 4),
        parang=parang,
        n_signal_times=n_signal_times,
        frame_size=(x_size, y_size),
        psf_template=psf_template,
        minimum_similarity=0.5,
    )
    assert np.isnan(signal_time)
    assert np.isnan(similarity)

    # Case 4
    signal_time, similarity = get_hypothesis_for_position(
        residuals=residuals,
        position=(8, 2),
        parang=parang,
        n_signal_times=n_signal_times,
        frame_size=(x_size, y_size),
        psf_template=psf_template,
        minimum_similarity=-1,
    )
    assert np.isnan(signal_time)
    assert similarity == 0


def test__get_all_hypotheses() -> None:
    """
    Test `hsr4hci.hypotheses.get_all_hypotheses`.
    """

    # Fix random seed
    np.random.seed(42)

    # Create fake PSF template
    x, y = np.meshgrid(np.arange(33), np.arange(33))
    model = models.Gaussian2D(x_mean=17, y_mean=17)
    psf_template = model(x, y)

    # Create test data
    n_frames, x_size, y_size = 100, 13, 13
    parang = np.linspace(0, 90, n_frames)
    n_signal_times = 5
    signal_times = get_signal_times(n_frames, n_signal_times)
    roi_mask = get_circle_mask((x_size, y_size), 6)
    residuals: Dict[str, np.ndarray] = {}
    for i, signal_time in enumerate(signal_times):

        # For all cases
        tmp_residuals = np.random.normal(0, 0.0001, (n_frames, x_size, y_size))
        tmp_residuals = tmp_residuals.astype(np.float32)
        tmp_residuals[:, ~roi_mask] = np.nan

        # For case 1
        if i == 2:
            tmp_residuals[:, 7, 7] = get_time_series_for_position(
                position=(7, 7),
                signal_time=signal_time,
                frame_size=(x_size, y_size),
                parang=parang,
                psf_template=psf_template,
            )
        if i == 3:
            tmp_residuals[:, 8, 2] = get_time_series_for_position(
                position=(8, 2),
                signal_time=signal_time,
                frame_size=(x_size, y_size),
                parang=parang,
                psf_template=psf_template,
            )
        if i == 4:
            tmp_residuals[:, 11, 5] = get_time_series_for_position(
                position=(11, 5),
                signal_time=signal_time,
                frame_size=(x_size, y_size),
                parang=parang,
                psf_template=psf_template,
            )

        residuals[str(signal_time)] = tmp_residuals

    # Case 1
    hypotheses, similarities = get_all_hypotheses(
        roi_mask=roi_mask,
        residuals=residuals,
        parang=parang,
        n_signal_times=n_signal_times,
        frame_size=(x_size, y_size),
        psf_template=psf_template,
        minimum_similarity=0.5,
        n_roi_splits=1,
        roi_split=0,
    )
    target_hypotheses = np.full_like(hypotheses, np.nan)
    target_similarities = np.full_like(similarities, np.nan)
    target_hypotheses[7, 7] = signal_times[2]
    target_hypotheses[8, 2] = signal_times[3]
    target_hypotheses[11, 5] = signal_times[4]
    target_similarities[7, 7] = 1
    target_similarities[8, 2] = 1
    target_similarities[11, 5] = 1
    assert np.array_equal(hypotheses, target_hypotheses, equal_nan=True)
    assert np.array_equal(similarities, target_similarities, equal_nan=True)
