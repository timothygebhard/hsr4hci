"""
Tests for general.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import pytest

from hsr4hci.utils.general import crop_center, \
    get_from_nested_dict, \
    set_in_nested_dict


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__crop_center() -> None:

    # Test case 0: Check what happens in dimensions don't match
    with pytest.raises(AssertionError) as error:
        array = np.array([1, 2, 3, 4, 5, 6, 7])
        crop_center(array=array, size=(1, 2, 3))
    assert 'Length of size must match number of dimensions' in str(error)

    # -------------------------------------------------------------------------
    # 1D case
    # -------------------------------------------------------------------------

    # Test case 1: 1D, odd array length, odd size
    array = np.array([1, 2, 3, 4, 5, 6, 7])
    result = crop_center(array=array, size=(3,))
    assert np.array_equal(result, np.array([3, 4, 5]))

    # Test case 2: 1D, even array length, odd size
    array = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    result = crop_center(array=array, size=(3,))
    assert np.array_equal(result, np.array([4, 5, 6]))

    # Test case 3: 1D, odd array length, even size
    array = np.array([1, 2, 3, 4, 5, 6, 7])
    result = crop_center(array=array, size=(4,))
    assert np.array_equal(result, np.array([2, 3, 4, 5]))

    # Test case 4: 1D, even array length, even size
    array = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    result = crop_center(array=array, size=(4,))
    assert np.array_equal(result, np.array([3, 4, 5, 6]))

    # Test case 5: 1D, arbitrary array length, size -1 (= do not crop)
    array = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    result = crop_center(array=array, size=(-1,))
    assert np.array_equal(result, array)

    # -------------------------------------------------------------------------
    # 2D case
    # -------------------------------------------------------------------------

    # Test case 6: 2D, (odd, odd)-array length, (odd, odd)-size
    array = np.arange(1, 26).reshape(5, 5)
    result = crop_center(array=array, size=(3, 3))
    assert np.array_equal(result, np.array([[7, 8, 9],
                                            [12, 13, 14],
                                            [17, 18, 19]]))


def test__get_from_nested_dict() -> None:

    dictionary = {'a': {'b': 42}}
    result = get_from_nested_dict(dictionary, ['a', 'b'])
    assert result == 42


def test__set_in_nested_dict() -> None:

    dictionary = {'a': {'b': 42}}
    set_in_nested_dict(dictionary, ['a', 'b'], 23)
    assert dictionary == {'a': {'b': 23}}
