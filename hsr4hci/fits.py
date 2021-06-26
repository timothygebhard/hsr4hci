"""
Utility functions for reading and writing FITS files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Optional, Tuple, Union, overload
from typing_extensions import Literal

import json

from astropy.io import fits

import numpy as np


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

@overload
def read_fits(
    file_path: Union[Path, str], return_header: Literal[True]
) -> Tuple[np.ndarray, dict]:
    ...  # pragma: no cover


@overload
def read_fits(
    file_path: Union[Path, str], return_header: Literal[False]
) -> np.ndarray:
    ...  # pragma: no cover


def read_fits(
    file_path: Union[Path, str], return_header: bool = False
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

    # If the array is boolean, convert to integer (FITS does not support bool)
    if array.dtype == 'bool':
        array = array.astype(int)

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
