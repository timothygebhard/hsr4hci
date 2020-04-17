"""
Tests for general.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from hsr4hci.utils.general import get_from_nested_dict, set_in_nested_dict


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__get_from_nested_dict():

    dictionary = {'a': {'b': 42}}
    result = get_from_nested_dict(dictionary, ['a', 'b'])
    assert result == 42


def test__set_in_nested_dict():

    dictionary = {'a': {'b': 42}}
    set_in_nested_dict(dictionary, ['a', 'b'], 23)
    assert dictionary == {'a': {'b': 23}}
