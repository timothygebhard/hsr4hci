"""
Utility functions for reading and writing FITS files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple, Type, Union

import json

from astropy.io import fits

import numpy as np


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def is_fits_file(file_path: Union[Path, str]) -> bool:
    """
    Check the ending of a file_path to determine if it is a FITS file.

    Note: checking for magic numbers to determine the file type does
        not work in general, because ESO FITS files seem to not always
        have the correct MIME type.

    Args:
        file_path: Path to the target file to be checked.

    Returns:
        True if the given file is a FITS file; False otherwise.
    """

    return Path(file_path).name.endswith('.fits')


def read_fits(
    file_path: Union[Path, str],
    return_header: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Open a FITS file and return its contents as a numpy array.

    Args:
        file_path: Path of the FITS file to be read in.
        return_header: Whether or not to return the FITS header.

    Returns:
        A numpy array containing the contents of the given FITS file.
    """

    # Make sure that file_path is a proper Path
    file_path = Path(file_path)

    # Open the FITS file and read the contents as well as the header
    with fits.open(file_path.as_posix()) as hdulist:
        array = np.array(hdulist[0].data)
        header = dict(hdulist[0].header)

    # Return either the contents and the header, or just the contents
    if return_header:
        return array, header
    return array


def save_fits(
    array: np.ndarray,
    file_path: Union[Path, str],
    header: Optional[dict] = None,
    overwrite: bool = True,
) -> None:
    """
    Save a numpy array as a FITS file (e.g., to inspect it with DS9).

    Args:
        array: The numpy array to be saved to a FITS file.
        file_path: The path where to save the FITS file.
        header: A dictionary with additional header information.
        overwrite: Whether or not to overwrite an existing FITS file.
    """

    # Make sure that file_path is a proper Path
    file_path = Path(file_path)

    # Create a new HDU for the array
    hdu = fits.PrimaryHDU(array)

    # If applicable, add header information
    if header is not None:
        for key, value in header.items():

            # FITS does not support list-type values in the header, which is
            # why these values need to be serialized to strings
            if isinstance(value, (list, tuple)):
                value = json.dumps(value)
            if isinstance(value, np.ndarray):
                value = json.dumps(value.tolist())

            # Take special care of NaN, because FITS can't deal with them
            if not isinstance(value, str) and np.isnan(value):
                value = 'NaN'

            # Save value to HDU header. We cast the key to all-caps because
            # that is the default for FITS; that is, header fields that are
            # automatically, such as NAXIS, are always all-caps.
            hdu.header[key.upper()] = value

    # Save the HDU to the specified FITS file
    fits.HDUList([hdu]).writeto(file_path.as_posix(), overwrite=overwrite)


def get_fits_header(file_path: Union[Path, str]) -> dict:
    """
    Open a FITS file and return only its header as a dictionary.

    Args:
        file_path: Path of the FITS file to be read in.

    Returns:
        A dictionary containing the header of the FITS file.
    """

    # Make sure that file_path is a proper Path
    file_path = Path(file_path)

    # Open the FITS file and read the contents as well as the header
    with fits.open(file_path.as_posix()) as hdulist:
        header = dict(hdulist[0].header)

    return header


def get_fits_header_value(
    file_path: Union[Path, str],
    key: str,
    dtype: Optional[Type] = None,
) -> Any:
    """
    Get the value of a field in the header of a given FITS file.

    Args:
        file_path: Path to the FITS file to read the header from.
        key: Key of the field in the header of the FITS file.
        dtype: Target data type of the header value. If this value is
            not None, we try to convert the value that was read from
            the FITS file to the given data type.

    Returns:
        The value of the header field in the given FITS file.
    """

    # Make sure that file_path is a proper Path
    file_path = Path(file_path)

    # Open the FITS file and read the target header value
    with fits.open(file_path.as_posix()) as hdu_list:
        value = hdu_list[0].header[key]

    # If desired, attempt to convert the value to the given data type
    if dtype is not None and isinstance(dtype, type):

        # Datetime objects in ISO 8601 format require special treatment.
        # Furthermore, we assume that all datetime values will be given in UTC.
        if dtype == datetime:
            value = datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%f')
            value = value.replace(tzinfo=timezone.utc)

        # Other values can be cast directly
        else:
            value = dtype(value)

    return value


def get_fits_header_value_array(
    file_path: Union[Path, str],
    start_key: str,
    end_key: str,
) -> np.ndarray:
    """
    For some fields, the ESO/VLT FITS files define a start and an end
    value (e.g., the seeing). This function will return a numpy array
    that has as many entries as there are frames in the FITS file, and
    that contains a linear interpolation between the start and the end
    value of the given keys.

    If the start and end key are identical, the resulting array will
    contain the same value everywhere.

    Args:
        file_path: Path to the FITS file to read the header from.
        start_key: The start key of the value we want to retrieve.
        end_key: The end key of the value we want to retrieve.

    Returns:
        A numpy array (that has as many entries as there are frames
        along NAXIS3 in the FITS file) containing a linear interpolation
        of the values corresponding to the given start and end keys.
    """

    # Make sure that file_path is a proper Path
    file_path = Path(file_path)

    # Get the number of frames in the current cube
    n_frames: int = get_fits_header_value(file_path, 'NAXIS3', int)

    # Read the start and end value from the FITS file
    start_value: float = get_fits_header_value(file_path, start_key, float)
    end_value: float = get_fits_header_value(file_path, end_key, float)

    # Create an array with a linear interpolation between those two values
    array = np.linspace(start_value, end_value, n_frames)

    return array


def header_value_exists(
    file_path: Union[Path, str],
    key: str,
) -> bool:
    """
    Check if the header of a given FITS file contains a given key.

    Args:
        file_path: Path to the FITS file to check.
        key: Name of the header field whose existence to check for.

    Returns:
        True if the header of the given FITS file contains the given
        key, and False otherwise.
    """

    # Make sure that file_path is a proper Path
    file_path = Path(file_path)

    # Open the FITS file and check if the requested key is present.
    # The following approach based on try/except is necessary because of a
    # strange convention regarding the role of the "HIERARCH" part of the
    # name of many ESO FITS files, which is best illustrated as follows:
    #
    #   >>> header["HIERARCH ESO TEL AMBI TAU0"]
    #   0.0053
    #   >>> header["ESO TEL AMBI TAU0"]
    #   0.0053
    #   >>> "HIERARCH ESO TEL AMBI TAU0" in header.keys()
    #   False
    #   >>> "ESO TEL AMBI TAU0" in header.keys()
    #   True
    #
    # Just because a key is not in `header.keys()` does therefore *not* mean
    # that we cannot access that key -- hence the try/except construct.
    with fits.open(file_path.as_posix()) as hdu_list:
        try:
            _ = hdu_list[0].header[key]
            return True
        except KeyError:
            return False
