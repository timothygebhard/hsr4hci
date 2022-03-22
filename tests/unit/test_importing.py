"""
Tests for importing.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

from hsr4hci.importing import get_member_by_name


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__get_member_by_name() -> None:
    """
    Test `hsr4hci.importing.get_member_by_name`.
    """

    member = get_member_by_name(module_name='pathlib', member_name='Path')
    assert member is Path
