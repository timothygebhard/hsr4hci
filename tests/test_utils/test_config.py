"""
Tests for config.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os

import pytest

from hsr4hci.utils.config import get_data_dir


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__get_data_dir(monkeypatch):

    # NOTE: We use the `monkeypatch` fixture of pytest [1] here to safely
    # modify and delete the value of environmental variables --- meaning that
    # even if the test fails, the variables will be restored to their original
    # values as to not affect any future tests that may also depend on them.
    # [1]: https://docs.pytest.org/en/latest/monkeypatch.html#monkeypatching-environment-variables

    # Case 1: HSR4HCI_DATA_DIR contains a "valid" data directory
    monkeypatch.setenv("HSR4HCI_DATA_DIR", ".")
    assert get_data_dir() == '.'

    # Case 2: HSR4HCI_DATA_DIR is not set
    monkeypatch.delenv("HSR4HCI_DATA_DIR", raising=False)
    with pytest.raises(RuntimeError) as error:
        get_data_dir()
    assert str(error.value) == 'Environment variable HSR4HCI_DATA_DIR not set!'

    # Case 3: HSR4HCI_DATA_DIR points to something that is not a directory
    monkeypatch.setenv("HSR4HCI_DATA_DIR", os.path.realpath(__file__))
    with pytest.raises(RuntimeError) as error:
        get_data_dir()
    assert str(error.value) == 'Value of HSR4HCI_DATA_DIR is not a directory!'
