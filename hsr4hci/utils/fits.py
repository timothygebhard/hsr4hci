"""
Functions for reading and writing FITS files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

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
              overwrite: bool = True):
    """
    Save a numpy array as a FITS file (e.g., to inspect it with DS9).

    Args:
        array: The numpy array to be saved to a FITS file.
        file_path: The path where to save the FITS file.
        overwrite: Whether or not to overwrite an existing FITS file.
    """

    hdu = fits.PrimaryHDU(array)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(file_path, overwrite=overwrite)
