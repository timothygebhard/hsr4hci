"""
Tests for masking.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from astropy.modeling import models
from astropy.units import Quantity

import numpy as np
import pytest

from hsr4hci.masking import (
    get_annulus_mask,
    get_circle_mask,
    get_exclusion_mask,
    get_positions_from_mask,
    get_predictor_mask,
    get_roi_mask,
    get_predictor_pixel_selection_mask,
    remove_connected_components,
)


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def psf_template() -> np.ndarray:

    x, y = np.meshgrid(np.arange(33), np.arange(33))
    gaussian = models.Gaussian2D(
        x_mean=17, x_stddev=1, y_mean=17, y_stddev=1, amplitude=1
    )
    psf_template = np.asarray(gaussian(x, y))
    psf_template /= np.max(psf_template)

    return psf_template


def test__get_circle_mask() -> None:

    # Case 1
    circle_mask = get_circle_mask(mask_size=(3, 3), radius=1, center=None)
    assert np.array_equal(
        circle_mask, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    )

    # Case 2
    circle_mask = get_circle_mask(mask_size=(5, 5), radius=2, center=None)
    assert np.array_equal(
        circle_mask,
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
    )

    # Case 3
    circle_mask = get_circle_mask(mask_size=(9, 9), radius=4, center=None)
    assert np.array_equal(
        circle_mask,
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
    )

    # Case 4
    circle_mask = get_circle_mask(mask_size=(5, 5), radius=2, center=(1, 1))
    assert np.array_equal(
        circle_mask,
        np.array(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
    )

    # Case 5
    circle_mask = get_circle_mask(mask_size=(4, 4), radius=1, center=(2, 2))
    assert np.array_equal(
        circle_mask,
        np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
            ]
        ),
    )


def test__get_annulus_mask() -> None:

    # Case 1
    annulus_mask = get_annulus_mask(
        mask_size=(3, 3), inner_radius=0, outer_radius=1, center=None
    )
    assert np.array_equal(
        annulus_mask,
        np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ]
        ),
    )

    # Case 2
    annulus_mask = get_annulus_mask(
        mask_size=(5, 5), inner_radius=1, outer_radius=2, center=None
    )
    assert np.array_equal(
        annulus_mask,
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        ),
    )

    # Case 3
    annulus_mask = get_annulus_mask(
        mask_size=(5, 5), inner_radius=1, outer_radius=3, center=(3, 3)
    )
    assert np.array_equal(
        annulus_mask,
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 1, 1, 0, 1],
                [0, 1, 1, 1, 1],
            ]
        ),
    )


def test__get_roi_mask() -> None:

    roi_mask = get_roi_mask(
        mask_size=(51, 51),
        inner_radius=Quantity(4, 'pixel'),
        outer_radius=Quantity(14, 'pixel'),
    )
    annulus_mask = get_annulus_mask(
        mask_size=(51, 51), inner_radius=4, outer_radius=14, center=None
    )
    assert np.array_equal(roi_mask, annulus_mask)


def test__get_predictor_mask() -> None:

    # Case 1
    predictor_mask = get_predictor_mask(
        mask_size=(11, 11),
        position=(3, 3),
        annulus_width=Quantity(0, 'pixel'),
        radius_position=Quantity(2, 'pixel'),
        radius_mirror_position=Quantity(1, 'pixel'),
    )
    assert np.array_equal(
        predictor_mask,
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
    )

    # Case 2
    predictor_mask = get_predictor_mask(
        mask_size=(11, 11),
        position=(5, 2),
        annulus_width=Quantity(1, 'pixel'),
        radius_position=Quantity(3, 'pixel'),
        radius_mirror_position=Quantity(3, 'pixel'),
    ).astype(int)
    assert np.array_equal(
        predictor_mask,
        np.array(
            [
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            ]
        ),
    )


def test__get_exclusion_mask(psf_template: np.ndarray) -> None:

    # Case 1
    with pytest.raises(ValueError) as value_error:
        get_exclusion_mask(
            mask_size=(11, 11),
            position=(5, 5),
            parang=np.linspace(0, 90, 200),
            psf_template=np.ones((1, 1)),
            signal_time=-1,
        )
    assert 'Negative signal times are not allowed!' in str(value_error)

    # Case 2
    exclusion_mask = get_exclusion_mask(
        mask_size=(11, 11),
        position=(5, 5),
        parang=np.linspace(0, 90, 200),
        psf_template=np.ones((1, 1)),
        signal_time=None,
    )
    assert np.sum(exclusion_mask) == 13

    # Case 3
    exclusion_mask = get_exclusion_mask(
        mask_size=(101, 101),
        position=(32, 21),
        parang=np.linspace(0, 90, 200),
        psf_template=psf_template,
        signal_time=None,
    )
    assert np.sum(exclusion_mask) == 57

    # Case 4
    exclusion_mask = get_exclusion_mask(
        mask_size=(101, 101),
        position=(32, 21),
        parang=np.linspace(0, 120, 200),
        psf_template=psf_template,
        signal_time=100,
    )
    assert np.sum(exclusion_mask) == 55

    # Case 4
    exclusion_mask = get_exclusion_mask(
        mask_size=(101, 101),
        position=(15, 15),
        parang=np.linspace(0, 90, 50),
        psf_template=psf_template,
        signal_time=20,
    )
    assert np.sum(exclusion_mask) == 45


def test__get_predictor_pixel_selection_mask(psf_template: np.ndarray) -> None:

    mask = get_predictor_pixel_selection_mask(
        mask_size=(101, 101),
        position=(15, 19),
        signal_time=20,
        parang=np.linspace(20, 120, 50),
        annulus_width=Quantity(0, 'pixel'),
        radius_position=Quantity(3, 'pixel'),
        radius_mirror_position=Quantity(3, 'pixel'),
        psf_template=psf_template,
    )
    assert np.sum(mask) == 28


def test__get_positions_from_mask() -> None:

    mask = np.full((11, 11), False)
    mask[3, 7] = True
    mask[2, 8] = True

    assert get_positions_from_mask(mask) == [(2, 8), (3, 7)]


def test__remove_connected_components() -> None:

    mask_1 = get_circle_mask(mask_size=(101, 101), radius=1, center=(30, 30))
    mask_2 = get_circle_mask(mask_size=(101, 101), radius=2, center=(36, 36))
    mask_3 = get_circle_mask(mask_size=(101, 101), radius=3, center=(50, 50))
    mask = np.asarray(sum([mask_1, mask_2, mask_3])).astype(bool)

    assert np.allclose(remove_connected_components(mask, None, None), mask)
    assert np.allclose(
        remove_connected_components(mask, 30, None), np.zeros_like(mask)
    )
    assert np.allclose(
        remove_connected_components(mask, None, 0), np.zeros_like(mask)
    )
    assert np.allclose(remove_connected_components(mask, 5, 20), mask_2)

    with pytest.raises(ValueError) as error:
        remove_connected_components(np.random.normal(0, 1, (5, 5)), None, None)
    assert 'Input image must be binary!' in str(error)
