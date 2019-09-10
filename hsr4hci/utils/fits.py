"""
Functions for reading and writing FITS files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Optional

from astropy.io import fits

import numpy as np


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def read_fits(file_path: str) -> np.ndarray:
    """
    Open a FITS file and return its contents as a numpy array.

    Args:
        file_path: Path of the FITS file to be read in.

    Returns:
        A numpy array containing the contents of the given FITS file.
    """

    with fits.open(file_path) as hdulist:
        array = np.array(hdulist[0].data)

    return array


def save_fits(array: np.ndarray,
              file_path: str,
              header: Optional[dict] = None,
              overwrite: bool = True):
    """
    Save a numpy array as a FITS file (e.g., to inspect it with DS9).

    Args:
        array: The numpy array to be saved to a FITS file.
        file_path: The path where to save the FITS file.
        header: A dictionary with additional header information.
        overwrite: Whether or not to overwrite an existing FITS file.
    """

    # Create a new HDU for the array
    hdu = fits.PrimaryHDU(array)

    # If applicable, add header information
    if header is not None:
        for key, value in header.items():

            # Take special care of NaN, because FITS can't deal with them
            if isinstance(value, (list, tuple, np.ndarray)):
                value = tuple(['NaN' if np.isnan(_) else _ for _ in value])
            else:
                value = 'NaN' if np.isnan(value) else value

            # Save value to HDU header
            hdu.header[key] = value

    # Save the HDU to the specified FITS file
    fits.HDUList([hdu]).writeto(file_path, overwrite=overwrite)
