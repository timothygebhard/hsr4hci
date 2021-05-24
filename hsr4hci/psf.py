"""
Utility functions for working with point spread functions.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any

from astropy.modeling import models, fitting

import numpy as np

from hsr4hci.coordinates import get_center
from hsr4hci.general import crop_center


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_psf_fwhm(psf_template: np.ndarray) -> float:
    """
    Fit a symmetric 2D Gauss function to the given `psf_template` to
    estimate the full width half maximum (FWHM) of the central "blob".

    Args:
        psf_template: A 2D numpy array containing the raw, unsaturated
            PSF template.

    Returns:
        The FWHM of the PSF template (in pixels).
    """

    # Crop PSF template: too large templates (which are mostly zeros) can
    # cause problems when fitting them with a 2D Gauss function
    psf_cropped = np.copy(psf_template)
    if psf_template.shape[0] >= 33 or psf_template.shape[1] >= 33:
        psf_cropped = crop_center(psf_cropped, (33, 33))

    # Define the grid for the fit
    x, y = np.meshgrid(
        np.arange(psf_cropped.shape[0]), np.arange(psf_cropped.shape[1])
    )

    # Create a new Gaussian2D object
    center = get_center(psf_cropped.shape)
    gaussian = models.Gaussian2D(x_mean=center[0], y_mean=center[1])

    # Define auxiliary function for tieing the standard deviations
    def tie_stddev(gaussian: Any) -> Any:
        return gaussian.y_stddev

    # Enforce symmetry: tie standard deviation parameters to same value to
    # ensure that the resulting 2D Gaussian is always circular
    gaussian.x_stddev.tied = tie_stddev

    # Fix the position (= mean) of the 2D Gaussian
    gaussian.x_mean.fixed = True
    gaussian.y_mean.fixed = True

    # Fit the model to the data
    fit_p = fitting.LevMarLSQFitter()
    gaussian_model = fit_p(gaussian, x, y, np.nan_to_num(psf_cropped))

    # Make sure the returned FWHM is positive
    return abs(float(gaussian_model.x_fwhm))
