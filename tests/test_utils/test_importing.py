"""
Tests for importing.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

from hsr4hci.utils.importing import get_member_by_name


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__get_member_by_name() -> None:

    member = get_member_by_name(module_name='pathlib',
                                member_name='Path')
    assert member is Path
