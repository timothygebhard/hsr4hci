"""
Tests for residuals.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from astropy.modeling import models

import numpy as np
import pytest

# noinspection PyProtectedMember
from hsr4hci.residuals import (
    _get_expected_signal,
    _prune_blobs,
    assemble_residual_stack_from_hypotheses,
    get_residual_selection_mask,
)

from hsr4hci.general import crop_or_pad, shift_image
from hsr4hci.masking import get_circle_mask


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__assemble_residuals_from_hypotheses() -> None:
    """
    Test `hsr4hci.residuals.assemble_residual_stack_from_hypotheses`.
    """

    hypotheses = np.array([[1, 2, 4], [4, 1, 2], [3, 4, 1]])
    selection_mask = np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]]).astype(bool)
    residuals = {
        'default': np.full((5, 3, 3), 0),
        '1': np.full((5, 3, 3), 1),
        '2': np.full((5, 3, 3), 2),
        '3': np.full((5, 3, 3), 3),
        '4': np.full((5, 3, 3), 4),
    }

    assembled = assemble_residual_stack_from_hypotheses(
        hypotheses=hypotheses,
        selection_mask=selection_mask,
        residuals=residuals,
    )
    assert np.allclose(
        assembled,
        np.tile(np.array([[0, 2, 4], [0, 1, 2], [0, 4, 1]]), (5, 1, 1)),
    )


def test___get_expected_signal() -> None:
    """
    Test `hsr4hci.residuals._get_expected_signal`.
    """

    x, y = np.meshgrid(np.arange(33), np.arange(33))
    model = models.Gaussian2D(x_mean=17, x_stddev=2, y_mean=17, y_stddev=2)
    psf_template = model(x, y)

    # Case 1
    expected_signal = _get_expected_signal(
        frame_size=(101, 101),
        field_rotation=360,
        psf_template=psf_template,
        grid_size=128,
        relative_rho=0.200,
        relative_phi=0.275,
    )
    assert np.argmax(np.mean(expected_signal, axis=1)) == 63
    assert np.isclose(np.sum(expected_signal), 1242.3824447431612)

    # Case 2
    expected_signal = _get_expected_signal(
        frame_size=(101, 101),
        field_rotation=90,
        psf_template=psf_template,
        grid_size=256,
        relative_rho=0.500,
        relative_phi=0.275,
    )
    assert np.argmax(np.mean(expected_signal, axis=0)) == 123
    assert np.argmax(np.mean(expected_signal, axis=1)) == 127
    assert np.isclose(np.sum(expected_signal), 1315.9877084014645)


def test___prune_blobs() -> None:
    """
    Test `hsr4hci.residuals._prune_blobs`.
    """

    # Case 1
    blobs = [
        (10.2, 90.0, 12),
        (11.4, 120.0, 14),
        (19.5, 0.0, 5.0),
    ]
    pruned = _prune_blobs(blobs)
    assert np.array_equal(pruned, np.array([[11.4, 120], [19.5, 0]]))


def test__get_residual_selection_mask() -> None:
    """
    Test `hsr4hci.residuals.get_residual_selection_mask`.
    """

    # Create fake PSF template
    x, y = np.meshgrid(np.arange(33), np.arange(33))
    model = models.Gaussian2D(x_mean=17, x_stddev=2, y_mean=17, y_stddev=2)
    psf_template = model(x, y)

    # Define shortcuts
    frame_size = (101, 101)
    parang = np.linspace(0, 90, 100)
    field_rotation = abs(parang[-1] - parang[0])
    psf_resized = crop_or_pad(psf_template, frame_size)

    # Create fake match fraction
    roi_mask = get_circle_mask(frame_size, 45)
    match_fraction = np.zeros(frame_size)
    match_fraction[~roi_mask] = np.nan
    planet_traces = np.zeros(frame_size)
    for (rho, phi) in [(16, 150)]:
        alpha = np.deg2rad(field_rotation / 2)
        for offset in np.linspace(-alpha, alpha, 100):
            x_ = float(rho * np.cos(np.deg2rad(phi) + offset))
            y_ = float(rho * np.sin(np.deg2rad(phi) + offset))
            shifted = shift_image(psf_resized, (x_, y_))
            planet_traces += shifted
    planet_traces /= np.max(planet_traces)
    match_fraction += planet_traces

    # Case 1
    with pytest.raises(RuntimeError) as runtime_error:
        get_residual_selection_mask(
            match_fraction=match_fraction,
            parang=np.linspace(0, 200, 100),
            psf_template=psf_template,
            grid_size=256,
        )
    assert 'Field rotation is greater than 180 degrees!' in str(runtime_error)

    # Case 2
    (
        selection_mask,
        polar,
        matched,
        expected_signal,
        positions,
    ) = get_residual_selection_mask(
        match_fraction=match_fraction,
        parang=parang,
        psf_template=psf_template,
        grid_size=128,
    )
    assert np.sum(selection_mask) == 207
    assert np.isclose(np.sum(polar), 110.40989835080471)
    assert np.isclose(np.sum(matched), 1.1368683772161603e-13)
    assert np.isclose(np.sum(np.clip(matched, 0, None)), 439.39300124870806)
    assert len(positions) == 1
    assert np.isclose(positions[0][0], 15.859616957913861)
    assert np.isclose(positions[0][1], 2.5572610861465055)
