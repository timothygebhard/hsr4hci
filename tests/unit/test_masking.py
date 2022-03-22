"""
Tests for masking.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from itertools import combinations

from astropy.units import Quantity

import numpy as np
import pytest

from hsr4hci.masking import (
    get_annulus_mask,
    get_circle_mask,
    get_exclusion_mask,
    get_partial_roi_mask,
    get_positions_from_mask,
    get_predictor_mask,
    get_predictor_pixel_selection_mask,
    get_roi_mask,
    mask_frame_around_position,
    remove_connected_components,
)


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__get_circle_mask() -> None:
    """
    Test `hsr4hci.masking.get_circle_mask`.
    """

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
    """
    Test `hsr4hci.masking.get_annulus_mask`.
    """

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
    """
    Test `hsr4hci.masking.get_roi_mask`.
    """

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
    """
    Test `hsr4hci.masking.get_predictor_mask`.
    """

    # Case 1
    predictor_mask = get_predictor_mask(
        mask_size=(11, 11),
        position=(3, 3),
        radius_position=Quantity(2, 'pixel'),
        radius_opposite=Quantity(1, 'pixel'),
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
        radius_position=Quantity(3, 'pixel'),
        radius_opposite=Quantity(3, 'pixel'),
    ).astype(int)
    assert np.array_equal(
        predictor_mask,
        np.array(
            [
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            ]
        ),
    )


def test__get_exclusion_mask() -> None:
    """
    Test `hsr4hci.masking.get_exclusion_mask`.
    """

    # Case 1
    exclusion_mask = get_exclusion_mask(
        mask_size=(11, 11),
        position=(5, 5),
        radius_excluded=Quantity(0, 'pixel'),
    )
    assert np.sum(exclusion_mask) == 0

    # Case 2
    exclusion_mask = get_exclusion_mask(
        mask_size=(11, 11),
        position=(5, 5),
        radius_excluded=Quantity(1, 'pixel'),
    )
    assert np.sum(exclusion_mask) == 1

    # Case 3
    exclusion_mask = get_exclusion_mask(
        mask_size=(101, 101),
        position=(32, 21),
        radius_excluded=Quantity(2, 'pixel'),
    )
    assert np.sum(exclusion_mask) == 9

    # Case 4
    exclusion_mask = get_exclusion_mask(
        mask_size=(101, 101),
        position=(10, 80),
        radius_excluded=Quantity(3, 'pixel'),
    )
    assert np.sum(exclusion_mask) == 25


def test__get_predictor_pixel_selection_mask() -> None:
    """
    Test `hsr4hci.masking.get_predictor_pixel_selection_mask`.
    """

    # Case 1
    mask = get_predictor_pixel_selection_mask(
        mask_size=(101, 101),
        position=(15, 19),
        radius_position=Quantity(3, 'pixel'),
        radius_opposite=Quantity(3, 'pixel'),
        radius_excluded=Quantity(2, 'pixel'),
    )
    assert np.sum(mask) == 41

    # Case 2
    mask = get_predictor_pixel_selection_mask(
        mask_size=(101, 101),
        position=(15, 19),
        radius_position=Quantity(5, 'pixel'),
        radius_opposite=Quantity(5, 'pixel'),
        radius_excluded=Quantity(3, 'pixel'),
    )
    assert np.sum(mask) == 113


def test__get_positions_from_mask() -> None:
    """
    Test `hsr4hci.masking.get_positions_from_mask`.
    """

    mask = np.full((11, 11), False)
    mask[3, 7] = True
    mask[2, 8] = True

    assert get_positions_from_mask(mask) == [(2, 8), (3, 7)]


def test__get_partial_roi_mask() -> None:
    """
    Test `hsr4hci.masking.get_partial_roi_mask`.
    """

    roi_mask = get_circle_mask((101, 101), 45)
    partial_roi_masks = [
        get_partial_roi_mask(roi_mask, i, 10) for i in range(10)
    ]

    assert np.array_equal(roi_mask, np.nansum(partial_roi_masks, axis=0))
    for mask_1, mask_2 in combinations(partial_roi_masks, 2):
        assert np.sum(np.logical_and(mask_1, mask_2)) == 0


def test__remove_connected_components() -> None:
    """
    Test `hsr4hci.masking.remove_connected_components`.
    """

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


def test__mask_around_position() -> None:
    """
    Test `hsr4hci.masking.mask_around_position`.
    """

    frame = np.ones((17, 17))
    masked_frame = mask_frame_around_position(
        frame=frame, position=(5, 9), radius=3,
    )
    assert np.sum(masked_frame) == 25
    assert masked_frame[5, 9] == 0
    assert masked_frame[9, 5] == 1
    assert masked_frame[8, 4] == 1
