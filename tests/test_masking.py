"""
Tests for masking.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import pytest

from hsr4hci.masking import (
    get_circle_mask,
    get_positions_from_mask,
    remove_connected_components,
)


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__get_circle_mask() -> None:

    # Case 1
    circle_mask = get_circle_mask(mask_size=(3, 3), radius=1, center=None)
    assert np.allclose(
        circle_mask, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    )

    # Case 2
    circle_mask = get_circle_mask(mask_size=(5, 5), radius=2, center=None)
    assert np.allclose(
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
    assert np.allclose(
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
    assert np.allclose(
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
    assert np.allclose(
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
