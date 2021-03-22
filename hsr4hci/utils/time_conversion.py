"""
Utility functions for converting between Python datetimes, timestamps
and ISO 8061 strings.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Union

from datetime import datetime, timezone
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
        date_string: A datetime as a string (e.g., "2007-04-05T14:30").
        tzinfo: A timezone object. Usually, everything should be in UTC.

    Returns:
        A datetime object that matches the given `date_string`.
    """

    # Decode the date string if necessary (this is sometimes necessary when
    # strings are read directly from a PynPoint data base)
    if isinstance(date_string, np.bytes_):
        date_string = date_string.decode('utf-8')

    # Convert the date string to a datetime
    return parse(date_string).replace(tzinfo=tzinfo)


def date_string_to_timestamp(
    date_string: Union[str, np.bytes_],
    tzinfo: timezone = timezone.utc,
) -> float:
    """
    Convert a date_string (usually ISO 8061 for FITS files) to a UNIX
    timestamp (i.e., seconds since January 1, 1970).

    Args:
        date_string: A datetime as a string (e.g., "2007-04-05T14:30").
        tzinfo: A timezone object. Usually, everything should be in UTC.

    Returns:
        A timestamp (as a float) that matches the given `date_string`.
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
        include_timezone: Whether or not to include the time zone
            information in the string (e.g., "+00:00").

    Returns:
        An ISO 8061-formatted string that matches the given timestamp.
    """
    if include_timezone:
        return datetime.fromtimestamp(timestamp, tz=tzinfo).isoformat()
    return datetime.fromtimestamp(timestamp, tz=tzinfo).isoformat()[:-6]
