"""
Functions for reading and writing FITS files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from astropy.io import fits
import numpy as np

from typing import List


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


def save_fits_multiarray(arrays: List[np.ndarray],
                         array_names: List[str],
                         file_path: str,
                         overwrite: bool = True):
    """
    Save multiple (named) numpy array to the same FITS file (e.g., to
    inspect them with DS9 using the "File > Open as > Multiple Extension
    Frames" option).

    Args:
        arrays: List of numpy array to be saved to a FITS file.
        array_names: Names to give to the HDU (one for each array).
        file_path: The path where to save the FITS file.
        overwrite: Whether or not to overwrite an existing FITS file.
    """

    # Helper function for getting the HDU classes: The first HDU in the file
    # needs to be a PrimaryHDU; all subsequent ones must be of class ImageHDU
    def _get_hdu_class():
        yield fits.PrimaryHDU
        while True:
            yield fits.ImageHDU

    # For each array, add an HDU with its respective name to the list of HDUs
    hdulist = list()
    for hdu_class, array, name in zip(_get_hdu_class(), arrays, array_names):
        hdu = hdu_class(array)
        hdu.name = name
        hdulist.append(hdu)

    # Save the resulting HDUList to a FITS file
    fits.HDUList(hdulist).writeto(file_path, overwrite=overwrite)
