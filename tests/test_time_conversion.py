"""
Tests for time_conversion.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from datetime import timezone, datetime

import numpy as np

from hsr4hci.time_conversion import (
    date_string_to_datetime,
    date_string_to_timestamp,
    timestamp_to_date_string,
    timestamp_to_datetime,
)


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__date_string_to_datetime() -> None:

    result = date_string_to_datetime(
        date_string='2021-04-30T15:36:29+00:00',
        tzinfo=timezone.utc,
    )
    assert result == datetime(2021, 4, 30, 15, 36, 29, tzinfo=timezone.utc)

    date_string = np.array(['2021-04-30T15:36:29+00:00'], dtype=np.bytes_)[0]
    result = date_string_to_datetime(
        date_string=date_string,
        tzinfo=timezone.utc,
    )
    assert result == datetime(2021, 4, 30, 15, 36, 29, tzinfo=timezone.utc)


def test__date_string_to_timestamp() -> None:

    result = date_string_to_timestamp(
        date_string='2021-04-30T15:36:29+00:00',
        tzinfo=timezone.utc,
    )
    assert result == 1619796989


def test__timestamp_to_datetime() -> None:

    result = timestamp_to_datetime(
        timestamp=1619796989,
        tzinfo=timezone.utc,
    )
    assert result == datetime(2021, 4, 30, 15, 36, 29, tzinfo=timezone.utc)


def test__timestamp_to_date_string() -> None:

    result = timestamp_to_date_string(
        timestamp=1619796989,
        tzinfo=timezone.utc,
        include_timezone=True,
    )
    assert result == '2021-04-30T15:36:29+00:00'

    result = timestamp_to_date_string(
        timestamp=1619796989,
        tzinfo=timezone.utc,
        include_timezone=False,
    )
    assert result == '2021-04-30T15:36:29'