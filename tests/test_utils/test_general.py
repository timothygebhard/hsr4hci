"""
Tests for general.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from _pytest.tmpdir import TempPathFactory
from deepdiff import DeepDiff

import numpy as np
import pytest

from hsr4hci.utils.general import (
    crop_center,
    get_from_nested_dict,
    get_md5_checksum,
    prestack_array,
    rotate_position,
    set_in_nested_dict,
    split_into_n_chunks
)


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


def test__prestack_array() -> None:

    # Test case 0: stacking_factor = 1
    array = np.random.normal(0, 1, 10)
    output = prestack_array(
        array=array, stacking_factor=1, stacking_function=np.median
    )
    assert np.allclose(output, array)

    # Test case 1: 1D, median
    array = np.array([1, 3, 5, 7, 9, 11])
    target = np.array([2, 6, 10])
    output = prestack_array(
        array=array, stacking_factor=2, stacking_function=np.median
    )
    assert np.allclose(output, target)

    # Test case 2: 1D, mean
    array = np.array([1, 2, 3, 4, 5, 6])
    target = np.array([2, 5])
    output = prestack_array(
        array=array, stacking_factor=3, stacking_function=np.mean
    )
    assert np.allclose(output, target)

    # Test case 3: 1D, median, stacking factor does not fit array length
    array = np.array([1, 3, 5, 7, 9])
    target = np.array([2, 6, 9])
    output = prestack_array(
        array=array, stacking_factor=2, stacking_function=np.median
    )
    assert np.allclose(output, target)

    # Test case 4: 2D, median
    array = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    target = np.array([[2, 3], [6, 7]])
    output = prestack_array(
        array=array, stacking_factor=2, stacking_function=np.median
    )
    assert np.allclose(output, target)


def test__rotate_position() -> None:

    # Check for ValueError
    with pytest.raises(ValueError) as error:
        rotate_position(np.array([(1, 1), (0, 0)]), (0, 0), np.array([0, 1]))
    assert 'position and angle cannot both be arrays!' in str(error)

    # Run tests for the case where `position` is simply a tuple
    position = (10, 10)
    center = (0, 0)
    assert np.allclose(rotate_position(position, center, -90), (10, -10))
    assert np.allclose(rotate_position(position, center, 0), (10, 10))
    assert np.allclose(rotate_position(position, center, 45), (0, 14.14213562))
    assert np.allclose(rotate_position(position, center, 90), (-10, 10))
    assert np.allclose(rotate_position(position, center, 180), (-10, -10))
    assert np.allclose(rotate_position(position, center, 360), (10, 10))
    assert np.allclose(rotate_position(position, center, 720), (10, 10))

    # Run tests for the case where `position` is an array
    position = np.array([(10, 10), (0, 10), (10, 0)]).T
    expected = np.array([(10, -10), (10, 0), (0, -10)]).T
    assert np.allclose(rotate_position(position, center, -90), expected)

    # Run tests for the case where `angle` is an array
    position = (10, 10)
    angle = np.array([0, 90, 180])
    expected = np.array([(10, 10), (-10, 10), (-10, -10)]).T
    assert np.allclose(rotate_position(position, center, angle), expected)


def test__get_md5_checksum(tmp_path_factory: TempPathFactory) -> None:

    # Define location of test file in temporary directory
    test_dir = tmp_path_factory.mktemp('general', numbered=False)
    file_path = test_dir / 'dummy.txt'

    # Create dummy file with known checksum
    with open(file_path, 'w') as text_file:
        text_file.write('This is a dummy file.')

    assert get_md5_checksum(file_path) == 'f30f68c2ec607099c3ade01dba0571d8'


def test__split_into_n_chunks() -> None:

    # Test case 0a: Check that an error is raised when len(sequence) < n_chunks
    with pytest.raises(ValueError) as error:
        sequence_0 = (0, 1)
        split_into_n_chunks(sequence=sequence_0, n_chunks=3)
    assert 'n_chunks is greater than len(sequence):' in str(error)

    # Test case 0b: Check that an error is raised when n_chunks <= 0
    with pytest.raises(ValueError) as error:
        sequence_0 = (0, 1)
        split_into_n_chunks(sequence=sequence_0, n_chunks=0)
    assert 'n_chunks must be a positive integer!' in str(error)

    # Test case 1: tuple, len(sequence) mod n_chunks == 0
    sequence_1 = (1, 2, 3, 4, 5, 6)
    target_1 = [(1, 2), (3, 4), (5, 6)]
    output_1 = split_into_n_chunks(sequence=sequence_1, n_chunks=3)
    assert not DeepDiff(output_1, target_1)

    # Test case 2: list, len(sequence) mod n_chunks != 0
    sequence_2 = [1, 2, 3, 4, 5]
    target_2 = [[1, 2], [3, 4], [5]]
    output_2 = split_into_n_chunks(sequence=sequence_2, n_chunks=3)
    assert not DeepDiff(output_2, target_2)

    # Test case 1: np.ndarray, len(sequence) mod n_chunks == 0
    sequence_3 = np.array(['a', 'b', 'c', 'd'])
    target_3 = [np.array(['a', 'b']), np.array(['c', 'd'])]
    output_3 = split_into_n_chunks(sequence=sequence_3, n_chunks=2)
    assert not DeepDiff(output_3, target_3)
