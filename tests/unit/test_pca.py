"""
Tests for pca.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import pytest

from hsr4hci.masking import get_circle_mask
from hsr4hci.pca import get_pca_signal_estimates


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__get_pca_signal_estimates() -> None:
    """
    Test `hsr4hci.pca.get_pca_signal_estimates`.
    """

    # Case 0
    with pytest.raises(ValueError) as value_error:
        get_pca_signal_estimates(
            stack=np.random.normal(0, 1, (10, 5, 5)),
            parang=np.linspace(0, 90, 10),
            n_components=np.array([2.0]),
            return_components=True,
        )
    assert 'n_components must be a (sequence of)' in str(value_error)

    # Case 1
    # Numbers are taken from section 3.1.1 of:
    # http://faculty.salina.k-state.edu/tim/DAT/_downloads/
    #   Principal_Component_Analysis_Tutorial.pdf
    stack = np.array(
        [
            [
                [-1.63, 0, 0],
                [0, 0, -0.63],
                [0, 0, 0],
            ],
            [
                [-1.63, 0, 0],
                [0, 0, -1.63],
                [0, 0, 0],
            ],
            [
                [-0.63, 0, 0],
                [0, 0, -0.63],
                [0, 0, 0],
            ],
            [
                [-2.63, 0, 0],
                [0, 0, -0.63],
                [0, 0, 0],
            ],
            [
                [2.38, 0, 0],
                [0, 0, 0.38],
                [0, 0, 0],
            ],
            [
                [1.38, 0, 0],
                [0, 0, 1.38],
                [0, 0, 0],
            ],
            [
                [2.38, 0, 0],
                [0, 0, 1.38],
                [0, 0, 0],
            ],
            [
                [0.38, 0, 0],
                [0, 0, 0.38],
                [0, 0, 0],
            ],
        ]
    )
    signal_estimates, components = get_pca_signal_estimates(
        stack=stack,
        parang=np.linspace(0, 90, 8),
        n_components=2,
        return_components=True,
        roi_mask=None,
    )
    assert np.allclose(
        components[0],
        np.array([[0.8950061, 0, 0], [0, 0, 0.4460539], [0, 0, 0]]),
    )
    assert np.allclose(
        components[1],
        np.array([[0.4460539, 0, 0], [0, 0, -0.8950061], [0, 0, 0]]),
    )

    # Case 3
    with pytest.warns(UserWarning, match=r'.*(n_components cannot be).*'):
        get_pca_signal_estimates(
            stack=np.random.normal(0, 1, (10, 5, 5)),
            parang=np.linspace(0, 90, 10),
            n_components=list(range(12)),
            return_components=True,
        )

    # Case 4
    np.random.seed(42)
    roi_mask = get_circle_mask(mask_size=(21, 21), radius=6)
    signal_estimates, components = get_pca_signal_estimates(
        stack=np.random.normal(0, 1, (17, 21, 21)),
        parang=np.linspace(0, 90, 17),
        n_components=5,
        return_components=True,
        roi_mask=roi_mask,
    )
    assert np.isnan(signal_estimates[:, ~roi_mask]).all()
    assert np.isnan(components[:, ~roi_mask]).all()
    assert np.isclose(np.nansum(signal_estimates), 0.06645434286164359)
    assert np.isclose(np.nansum(components), -0.4697774127481366)

    # Case 4
    signal_estimates = get_pca_signal_estimates(
        stack=np.random.normal(0, 1, (17, 11, 11)),
        parang=np.linspace(0, 90, 17),
        n_components=1,
        return_components=False,
        roi_mask=None,
    )
    assert np.isclose(np.nansum(signal_estimates), -0.4284529435116958)
