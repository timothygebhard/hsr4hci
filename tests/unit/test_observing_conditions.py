"""
Tests for observing_conditions.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from hsr4hci.general import prestack_array
from hsr4hci.observing_conditions import (
    get_observing_conditions,
    interpolate_observing_conditions,
    ObservingConditions,
    query_archive,
    resolve_parameter_name,
)


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------


def test__observing_conditions() -> None:

    # Case 1: illegal name for observing condition
    input_dict = {'all': np.random.normal(0, 1, 10)}
    with pytest.raises(KeyError) as key_error:
        ObservingConditions(input_dict)
    assert 'Illegal name: "all"!' in str(key_error)

    # Case 2: observing conditions are not numpy arrays
    input_dict = {
        'a': np.random.normal(0, 1, 10),
        'b': list(range(100)),  # type: ignore
    }
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


def test__query_archive() -> None:

    # Case 1
    with pytest.raises(ValueError) as value_error:
        query_archive(
            start_date='2016-04-10T00:00:00',
            end_date='2016-04-10T00:00:05',
            archive='illegal',
            parameter_key='wind_speedu',
        )
    assert 'Invalid archive' in str(value_error)

    # Case 2
    df = query_archive(
        start_date='2016-04-10T00:00:00',
        end_date='2016-04-10T00:00:05',
        archive='meteo',
        parameter_key='wind_speedu',
    )
    assert len(df) == 1
    assert df.loc[0]['integration_time'] == 60
    assert df.loc[0]['wind_speedu'] == -3.63
    assert df.loc[0]['timestamp'] == 1460246403

    # Case 3
    df = query_archive(
        start_date='2015-04-10T00:00:00',
        end_date='2015-04-10T00:10:000',
        archive='dimm_old',
        parameter_key='airmass',
    )
    assert len(df) == 3
    assert df.loc[0]['integration_time'] == 57
    assert df.loc[1]['airmass'] == 1.7
    assert df.loc[2]['timestamp'] == 1428624572

    # Case 4
    df = query_archive(
        start_date='2018-06-07T00:00:00',
        end_date='2018-06-07T00:02:00',
        archive='dimm_new',
        parameter_key='fwhm',
    )
    assert len(df) == 1
    assert df.loc[0]['integration_time'] == 60
    assert df.loc[0]['fwhm'] == 0.575
    assert df.loc[0]['timestamp'] == 1528329652

    # Case 5
    df = query_archive(
        start_date='2018-06-07T00:00:00',
        end_date='2018-06-07T00:02:00',
        archive='mass',
        parameter_key='tau',
    )
    assert len(df) == 1
    assert df.loc[0]['integration_time'] == 62
    assert df.loc[0]['tau'] == 0.002541
    assert df.loc[0]['timestamp'] == 1528329652

    # Case 6
    df = query_archive(
        start_date='2018-06-07T00:00:00',
        end_date='2018-06-07T00:01:00',
        archive='lhatpro',
        parameter_key='irt0',
    )
    assert len(df) == 1
    assert df.loc[0]['platform'] == 'A'
    assert df.loc[0]['integration_time'] == 92
    assert df.loc[0]['irt0'] == -95.840
    assert df.loc[0]['timestamp'] == 1528329600

    # Case 7
    df = query_archive(
        start_date='2018-06-07T00:00:00',
        end_date='2018-06-07T00:01:00',
        archive='lhatpro_irt',
        parameter_key='irt',
    )
    assert len(df) == 8
    assert np.unique(df['platform'].values) == np.array(['A'])
    assert np.unique(df['integration_time'].values) == np.array([5])
    assert df.loc[0]['irt'] == -95.51
    assert df.loc[1]['irt'] == -96.02
    assert df.loc[0]['timestamp'] == 1528329624
    assert df.loc[1]['timestamp'] == 1528329629


def test__interpolate_observing_conditions() -> None:

    np.random.seed(423)

    # Simulate smooth values with high temporal resolution (30 minutes at 1 Hz)
    start_time = 1528329652
    end_time = 1528329652 + 1800
    kernel_size = 128
    kernel = np.ones(kernel_size) / kernel_size
    true_air_pressure = np.random.normal(740, 5, 1800 + 4 * kernel_size)
    true_air_pressure = np.convolve(true_air_pressure, kernel, mode='same')
    true_air_pressure = np.convolve(true_air_pressure, kernel, mode='same')
    true_air_pressure = true_air_pressure[2 * kernel_size : -2 * kernel_size]

    # Apply temporal binning with a factor of 60 (i.e., only keep 1-minute
    # averages) to create mock archival data
    archive_air_pressure = prestack_array(true_air_pressure, 60)
    archive_timestamps = np.linspace(
        start_time, end_time, len(archive_air_pressure) + 1
    )

    # Interpolate the observing conditions based on the archival values
    # Note: We do not evaluate on the full archival values to avoid edge
    # effects (hence the +-120 second)
    df = pd.DataFrame(
        {
            'air_pressure': archive_air_pressure,
            'timestamp': archive_timestamps[1:],
        }
    )
    offset = 120
    interpolated_timestamps = np.arange(start_time + offset, end_time - offset)
    interpolated_air_pressure = interpolate_observing_conditions(
        timestamps=interpolated_timestamps,
        df=df,
        parameter_key='air_pressure',
    )

    # Compute the relative error, that is, the different between the "true"
    # values and the ones that we obtained using our interpolation
    relative_error = (
        interpolated_air_pressure - true_air_pressure[offset:-offset]
    ) / true_air_pressure[offset:-offset]

    assert np.isclose(np.mean(relative_error), 6.547663183121372e-08)


def test__get_observing_conditions() -> None:

    timestamps = np.arange(1359681191, 1359693381)
    interpolated, query_results = get_observing_conditions(
        parameter_name='air_pressure', timestamps=timestamps
    )
    assert len(timestamps) == len(interpolated)


def test__resolve_parameter_name() -> None:

    # Case 1
    archive, _, _ = resolve_parameter_name(
        parameter_name='air_pressure',
        obs_date=datetime(2015, 4, 4, 12, 0, 0, 0, timezone.utc),
    )
    assert archive == 'meteo'

    # Case 2
    archive, _, _ = resolve_parameter_name(
        parameter_name='coherence_time',
        obs_date=datetime(2015, 4, 4, 12, 0, 0, 0, timezone.utc),
    )
    assert archive == 'dimm_old'
    archive, _, _ = resolve_parameter_name(
        parameter_name='coherence_time',
        obs_date=datetime(2017, 4, 4, 12, 0, 0, 0, timezone.utc),
    )
    assert archive == 'mass'

    # Case 3
    archive, _, _ = resolve_parameter_name(
        parameter_name='isoplanatic_angle',
        obs_date=datetime(2015, 4, 4, 12, 0, 0, 0, timezone.utc),
    )
    assert archive == 'dimm_old'
    archive, _, _ = resolve_parameter_name(
        parameter_name='isoplanatic_angle',
        obs_date=datetime(2017, 4, 4, 12, 0, 0, 0, timezone.utc),
    )
    assert archive == 'mass'

    # Case 4
    archive, _, _ = resolve_parameter_name(
        parameter_name='observatory_temperature',
        obs_date=datetime(2015, 4, 4, 12, 0, 0, 0, timezone.utc),
    )
    assert archive == 'meteo'

    # Case 5
    archive, _, _ = resolve_parameter_name(
        parameter_name='relative_humidity',
        obs_date=datetime(2015, 4, 4, 12, 0, 0, 0, timezone.utc),
    )
    assert archive == 'meteo'

    # Case 6
    archive, _, _ = resolve_parameter_name(
        parameter_name='seeing',
        obs_date=datetime(2015, 4, 4, 12, 0, 0, 0, timezone.utc),
    )
    assert archive == 'dimm_old'
    archive, _, _ = resolve_parameter_name(
        parameter_name='seeing',
        obs_date=datetime(2017, 4, 4, 12, 0, 0, 0, timezone.utc),
    )
    assert archive == 'dimm_new'

    # Case 7
    archive, _, _ = resolve_parameter_name(
        parameter_name='wind_speed_u',
        obs_date=datetime(2015, 4, 4, 12, 0, 0, 0, timezone.utc),
    )
    assert archive == 'meteo'

    # Case 8
    archive, _, _ = resolve_parameter_name(
        parameter_name='wind_speed_v',
        obs_date=datetime(2019, 4, 4, 12, 0, 0, 0, timezone.utc),
    )
    assert archive == 'meteo'

    # Case 9
    archive, _, _ = resolve_parameter_name(
        parameter_name='wind_speed_w',
        obs_date=datetime(2010, 4, 4, 12, 0, 0, 0, timezone.utc),
    )
    assert archive == 'meteo'

    # Case 10
    with pytest.raises(ValueError) as value_error:
        resolve_parameter_name(
            parameter_name='illegal',
            obs_date=datetime(2010, 4, 4, 12, 0, 0, 0, timezone.utc),
        )
    assert 'Invalid parameter_name' in str(value_error)
