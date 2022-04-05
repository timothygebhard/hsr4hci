"""
Methods for converting between Python datetimes, timestamps and
ISO 8061 strings.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Union

from datetime import datetime, timezone, timedelta
from dateutil.parser import parse

import numpy as np


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def date_string_to_datetime(
    date_string: Union[str, np.bytes_],
    tzinfo: timezone = timezone.utc,
) -> datetime:
    """
    Convert a datetime from string (usually ISO 8061 for FITS files) to
    a Python datetime object.

    Args:
        date_string: A datetime string (e.g., `"2007-04-05T14:30"`).
        tzinfo: A timezone object. Usually, everything should be in UTC.

    Returns:
        A ``datetime`` object that matches the given ``date_string``.
    """

    # Decode the date string if necessary (this is sometimes necessary when
    # strings are read directly from a PynPoint database)
    if isinstance(date_string, np.bytes_):
        date_string = date_string.decode('utf-8')

    # Convert the date string to a datetime
    return parse(date_string).replace(tzinfo=tzinfo)


def date_string_to_timestamp(
    date_string: Union[str, np.bytes_],
    tzinfo: timezone = timezone.utc,
) -> float:
    """
    Convert a ``date_string`` (usually ISO 8061 for FITS files) to a
    UNIX timestamp (i.e., seconds since January 1, 1970).

    Args:
        date_string: A datetime string (e.g., `"2007-04-05T14:30"`).
        tzinfo: A timezone object. Usually, everything should be in UTC.

    Returns:
        A timestamp (as a float) that matches the given ``date_string``.
    """
    return date_string_to_datetime(date_string, tzinfo=tzinfo).timestamp()


def timestamp_to_datetime(
    timestamp: float,
    tzinfo: timezone = timezone.utc,
) -> datetime:
    """
    Convert a given UNIX timestamp to a Python datetime object.

    Args:
        timestamp: A UNIX timestamp (as a float).
        tzinfo: A timezone object. Usually, everything should be in UTC.

    Returns:
        A Python datetime object that matches the given timestamp.
    """
    return datetime.fromtimestamp(timestamp, tz=tzinfo)


def timestamp_to_date_string(
    timestamp: float,
    tzinfo: timezone = timezone.utc,
    include_timezone: bool = False,
) -> str:
    """
    Convert a given UNIX timestamp to a ISO 8061-formatted string.

    Args:
        timestamp: A UNIX timestamp (as a float).
        tzinfo: A timezone object. Usually, everything should be in UTC.
        include_timezone: Whether to include the time zone information
            in the string (e.g., `"+00:00"`).

    Returns:
        An ISO 8061-formatted string that matches the given timestamp.
    """
    if include_timezone:
        return datetime.fromtimestamp(timestamp, tz=tzinfo).isoformat()
    return datetime.fromtimestamp(timestamp, tz=tzinfo).isoformat()[:-6]


def round_minutes(
    dt: datetime,
    direction: str,
    resolution: float = 5,
) -> datetime:
    """
    Auxiliary function to round the minutes of a given datetime ``dt``
    to the desired ``resolution`` (e.g., closest 5 minutes). Seconds
    and milliseconds are discarded.

    Args:
        dt: A ``datetime`` object.
        direction: Either `"up"` or `"down"` (direction of rounding).
        resolution: Resolution (round to closest ``resolution``
            minutes).

    Returns:
        The given datetime ``dt`` rounded to the target ``resolution``.
    """

    # Compute new value for minute
    offset = int(direction == 'up')
    new_minute = (dt.minute // resolution + offset) * resolution

    # Construct new datetime object
    new_dt = dt + timedelta(minutes=new_minute - dt.minute)
    new_dt = new_dt.replace(second=0, microsecond=0)

    return new_dt
