"""
Tests for general.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import pytest

from hsr4hci.general import (
    crop_center,
    crop_or_pad,
    fast_corrcoef,
    find_closest,
    flatten_nested_dict,
    get_from_nested_dict,
    pad_array_to_shape,
    prestack_array,
    rotate_position,
    set_in_nested_dict,
    shift_image,
)


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__crop_center() -> None:
    """
    Test `hsr4hci.general.crop_center`.
    """

    # Test case 0: Check what happens in dimensions don't match
    with pytest.raises(RuntimeError) as error:
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

    # Test case 6: 1D, new shape is bigger than old shape
    array = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    result = crop_center(array=array, size=(10,))
    assert np.array_equal(result, array)

    # -------------------------------------------------------------------------
    # 2D case
    # -------------------------------------------------------------------------

    # Test case 6: 2D, (odd, odd)-array length, (odd, odd)-size
    array = np.arange(1, 26).reshape(5, 5)
    result = crop_center(array=array, size=(3, 3))
    assert np.array_equal(
        result, np.array([[7, 8, 9], [12, 13, 14], [17, 18, 19]])
    )


def test__find_closest() -> None:
    """
    Test `hsr4hci.general.find_closest`.
    """

    sequence = [1, 2, 3, 4, 5]

    assert find_closest(sequence=sequence, value=3.141) == (2, 3)
    assert find_closest(sequence=sequence, value=0.123) == (0, 1)
    assert find_closest(sequence=sequence, value=6.282) == (4, 5)
    assert find_closest(sequence=sequence, value=2.999) == (2, 3)


def test__get_from_nested_dict() -> None:
    """
    Test `hsr4hci.general.get_from_nested_dict`.
    """

    dictionary = {'a': {'b': 42}}
    result = get_from_nested_dict(dictionary, ['a', 'b'])
    assert result == 42


def test__set_in_nested_dict() -> None:
    """
    Test `hsr4hci.general.set_in_nested_dict`.
    """

    dictionary = {'a': {'b': 42}}
    set_in_nested_dict(dictionary, ['a', 'b'], 23)
    assert dictionary == {'a': {'b': 23}}


def test__prestack_array() -> None:
    """
    Test `hsr4hci.general.prestack_array`.
    """

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
    """
    Test `hsr4hci.general.rotate_position`.
    """

    # Check for ValueError
    with pytest.raises(ValueError) as error:
        rotate_position(np.array([(1, 1), (0, 0)]), (0, 0), np.array([0, 1]))
    assert 'position and angle cannot both be arrays!' in str(error)

    # Run tests for the case where `position` is simply a tuple
    position_tuple = (10, 10)
    center = (0, 0)
    assert np.allclose(rotate_position(position_tuple, center, -90), (10, -10))
    assert np.allclose(rotate_position(position_tuple, center, 0), (10, 10))
    assert np.allclose(
        rotate_position(position_tuple, center, 45), (0, 14.14213562)
    )
    assert np.allclose(rotate_position(position_tuple, center, 90), (-10, 10))
    assert np.allclose(
        rotate_position(position_tuple, center, 180), (-10, -10)
    )
    assert np.allclose(rotate_position(position_tuple, center, 360), (10, 10))
    assert np.allclose(rotate_position(position_tuple, center, 720), (10, 10))

    # Run tests for the case where `position` is an array
    position_array = np.array([(10, 10), (0, 10), (10, 0)]).T
    expected = np.array([(10, -10), (10, 0), (0, -10)]).T
    assert np.allclose(rotate_position(position_array, center, -90), expected)

    # Run tests for the case where `angle` is an array
    position = (10, 10)
    angle = np.array([0, 90, 180])
    expected = np.array([(10, 10), (-10, 10), (-10, -10)]).T
    assert np.allclose(rotate_position(position, center, angle), expected)


def test__pad_array_to_shape() -> None:
    """
    Test `hsr4hci.general.pad_array_to_shape`.
    """

    # Case 1
    with pytest.raises(ValueError) as error:
        pad_array_to_shape(array=np.ones((3, 3)), shape=(4,))
    assert 'Dimensions of array and shape do not align!' in str(error)

    # Case 2
    padded = pad_array_to_shape(array=np.ones((3,)), shape=(4,))
    assert np.allclose(padded, np.array([1, 1, 1, 0]))

    # Case 3
    padded = pad_array_to_shape(array=np.ones((3,)), shape=(5,))
    assert np.allclose(padded, np.array([0, 1, 1, 1, 0]))

    # Case 4
    padded = pad_array_to_shape(array=np.ones((1, 2)), shape=(3, 4))
    assert np.allclose(
        padded, np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
    )

    # Case 5
    with pytest.raises(ValueError) as error:
        pad_array_to_shape(array=np.ones((3, 3)), shape=(4, 2))
    assert 'is smaller than current size' in str(error)


def test__crop_or_pad() -> None:
    """
    Test `hsr4hci.general.crop_or_pad`.
    """

    # Case 1 (crop)
    result = crop_or_pad(array=np.arange(25).reshape((5, 5)), size=(3, 3))
    assert np.allclose(
        result, np.array([[6, 7, 8], [11, 12, 13], [16, 17, 18]])
    )

    # Case 2 (pad)
    result = crop_or_pad(array=np.ones((1, 1)), size=(3, 3))
    assert np.allclose(result, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))

    # Case 3
    with pytest.raises(RuntimeError) as error:
        crop_or_pad(array=np.ones((5, 5)), size=(7, 3))
    assert 'Mixing of cropping and padding is not supported!' in str(error)


def test__shift_image() -> None:
    """
    Test `hsr4hci.general.shift_image`.
    """

    # Case 1
    with pytest.raises(ValueError) as error:
        shift_image(image=np.random.normal(0, 1, (2, 3, 4)), offset=(1, 1))
    assert 'Input image must be 2D!' in str(error)

    # Case 2
    with pytest.raises(ValueError) as error:
        shift_image(image=np.ones((3, 3)), offset=(1, 1), interpolation='abc')
    assert 'The value of interpolation must be one' in str(error)

    # Case 3
    image = np.random.normal(0, 1, (3, 3))
    shifted = shift_image(
        image=image,
        offset=(1, 0),
        interpolation='bilinear',
    )
    target = np.roll(image, shift=1, axis=1)
    target[:, 0] = 0
    assert np.allclose(shifted, target)

    # Case 4
    image = np.random.normal(0, 1, (3, 3))
    shifted = shift_image(
        image=image,
        offset=(0, 1),
        interpolation='spline',
    )
    target = np.roll(image, shift=1, axis=0)
    target[0, :] = 0
    assert np.allclose(shifted, target)


def test__flatten_nested_dict() -> None:
    """
    Test `hsr4hci.general.flatten_nested_dict`.
    """

    # Case 1
    d = {'a': 5, 'b': 'foo', 'c': 2.7}
    assert flatten_nested_dict(d) == d

    # Case 2
    d = {'a': 5, 'b': {'c': 2.7}}
    assert flatten_nested_dict(d) == {'a': 5, 'b_c': 2.7}

    # Case 3
    d = {'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y': 10}}, 'd': [1, 2, 3]}
    d_flat = {'a': 1, 'c_a': 2, 'c_b_x': 5, 'd': [1, 2, 3], 'c_b_y': 10}
    assert flatten_nested_dict(d) == d_flat


def test__fast_corrcoef() -> None:
    """
    Test `hsr4hci.general.fast_corrcoef`.
    """

    np.random.seed(42)

    # Case 1
    x = np.random.normal(0, 1, 10)
    y = np.random.normal(0, 1, 10)
    a = fast_corrcoef(x, y)
    b = np.corrcoef(x, y)[0, 1]
    assert np.isclose(a, b).all()

    # Case 2
    x = np.random.normal(0, 1, 100)
    y = np.random.normal(0, 1, 100)
    a = fast_corrcoef(x, y)
    b = np.corrcoef(x, y)[0, 1]
    assert np.isclose(a, b).all()

    # Case 3
    x = np.random.normal(0, 1, 1000)
    y = np.random.normal(0, 1, 1000)
    a = fast_corrcoef(x, y)
    b = np.corrcoef(x, y)[0, 1]
    assert np.isclose(a, b).all()

    # Case 4
    x = np.random.normal(0, 1, 10)
    y = x + np.random.normal(0, 0.5, 10)
    a = fast_corrcoef(x, y)
    b = np.corrcoef(x, y)[0, 1]
    assert np.isclose(a, b).all()
