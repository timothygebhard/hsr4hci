"""
Tests for observing_conditions.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pytest

from hsr4hci.observing_conditions import ObservingConditions


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------


def test_observing_conditions() -> None:

    # Case 1: illegal name for observing condition
    input_dict = {'all': np.random.normal(0, 1, 10)}
    with pytest.raises(KeyError) as key_error:
        ObservingConditions(input_dict)
    assert 'Illegal name: "all"!' in str(key_error)

    # Case 2: observing conditions are not numpy arrays
    input_dict = {'a': np.random.normal(0, 1, 10), 'b': list(range(100))}
    with pytest.raises(ValueError) as value_error:
        ObservingConditions(input_dict)
    assert 'All observing conditions must be numpy arrays!' in str(value_error)

    # Case 3: observing conditions do not all have the same length
    input_dict = {
        'air_pressure': np.random.normal(0, 1, 11),
        'wind_speed_u': np.random.normal(0, 1, 10),
    }
    with pytest.raises(ValueError) as value_error:
        ObservingConditions(input_dict)
    assert 'All arrays must have the same length!' in str(value_error)

    # Create data for remaining test cases
    input_dict = {
        'air_pressure': np.random.normal(0, 1, 10),
        'wind_speed_u': np.random.normal(0, 1, 10),
    }
    observing_conditions = ObservingConditions(input_dict)

    # Case 4: test _verify_selected_keys()
    assert observing_conditions._verify_selected_keys([])
    assert observing_conditions._verify_selected_keys(None)
    assert observing_conditions._verify_selected_keys('all')
    assert observing_conditions._verify_selected_keys(['air_pressure'])
    assert observing_conditions._verify_selected_keys(
        ['air_pressure', 'wind_speed_u']
    )
    with pytest.raises(KeyError) as error:
        observing_conditions._verify_selected_keys('key_does_not_exist')
    assert 'is not a valid value for selected_keys' in str(error)

    # Case 5: test available_keys
    assert observing_conditions.available_keys == [
        'air_pressure',
        'wind_speed_u',
    ]

    # Case 6: test n_frames
    assert observing_conditions.n_frames == 10

    # Case 7: test as_dict()
    assert observing_conditions.as_dict(None) == {}
    assert observing_conditions.as_dict([]) == {}
    assert observing_conditions.as_dict('all') == input_dict

    # Case 8: test as_array()
    assert isinstance(observing_conditions.as_array(), np.ndarray)
    assert np.array_equal(
        observing_conditions.as_array(None), np.empty((10, 0))
    )
    assert observing_conditions.as_array('all').shape == (10, 2)

    # Case 8: test as_dataframe()
    assert isinstance(observing_conditions.as_dataframe(), pd.DataFrame)
    assert observing_conditions.as_dataframe().ndim == 2
    assert observing_conditions.as_dataframe(None).ndim == 2
    assert list(observing_conditions.as_dataframe('all').columns) == [
        'air_pressure',
        'wind_speed_u',
    ]
