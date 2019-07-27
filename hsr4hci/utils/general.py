"""
Functions useful in general.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from astropy.io import fits
import numpy as np

# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------


def save_as_fits(input_data: np.ndarray,
                 output_file: str,
                 overwrite: bool=False):

    hdu = fits.PrimaryHDU(input_data)
    hdul = fits.HDUList([hdu])
    hdul.writeto(output_file, overwrite=overwrite)
