"""
Utility functions for working with point spread functions.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from astropy.convolution import AiryDisk2DKernel
from astropy.units import Quantity

import numpy as np

from hsr4hci.fitting import CircularGauss2D
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
    if psf_template.shape[0] >= 33 and psf_template.shape[1] >= 33:
        psf_cropped = crop_center(psf_template, (33, 33))
    else:
        psf_cropped = psf_template

    # Define the grid for the fit
    x = np.arange(psf_cropped.shape[0])
    y = np.arange(psf_cropped.shape[1])
    meshgrid = (
        np.array(np.meshgrid(x, y)[0]),
        np.array(np.meshgrid(x, y)[1]),
    )

    # Set up a 2D Gaussian and fit it to the PSF template
    model = CircularGauss2D(
        mu_x=psf_cropped.shape[0] / 2 - 0.5,
        mu_y=psf_cropped.shape[1] / 2 - 0.5,
    )
    model.fit(meshgrid=meshgrid, target=psf_cropped)

    # Make sure the returned FWHM is positive
    return abs(model.fwhm)


def get_artificial_psf(
    pixscale: Quantity,
    lambda_over_d: Quantity,
) -> np.ndarray:
    """
    Create an artificial PSF template based on a 2D Airy function which
    can be used for data sets where no real PSF template is available.

    Args:
        pixscale: The PIXSCALE of the data set, usually in units of
            arc seconds per pixel.
        lambda_over_d: The ratio of the wavelength of the observation,
            lambda, and the size of the primary mirror of the telescope.
            Usually in units of arc seconds.

    Returns:
        A 2D numpy array containing an artificial PSF template.
    """

    # The factor of 1.383 is a "magic" number that was determined by
    # comparing real PSFs (for which the PIXSCALE and LAMBDA_OVER_D were
    # known) against the output of the AiryDisk2DKernel() function, and
    # adjusting the radius of the latter by a factor to minimize the
    # difference between the real and the fake PSF.
    return np.asarray(
        AiryDisk2DKernel(
            radius=1.383 * (lambda_over_d / pixscale).to('pixel').value,
        ).array
    )
