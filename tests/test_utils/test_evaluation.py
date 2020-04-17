"""
Tests for evaluation.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import pytest

from hsr4hci.utils.evaluation import TimeoutException, timeout_handler


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__timeout_handler():

    with pytest.raises(TimeoutException) as error:
        timeout_handler()

    assert str(error.value) == 'Optimization timed out!'
