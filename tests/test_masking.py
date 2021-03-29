"""
Tests for masking.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

from hsr4hci.masking import get_positions_from_mask


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__get_positions_from_mask() -> None:

    mask = np.full((11, 11), False)
    mask[3, 7] = True
    mask[2, 8] = True

    assert get_positions_from_mask(mask) == [(2, 8), (3, 7)]
