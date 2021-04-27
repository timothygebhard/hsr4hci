"""
Utility functions for working with point spread functions.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from astropy.convolution import AiryDisk2DKernel
from astropy.units import Quantity
from photutils import centroid_2dg, CircularAperture

import numpy as np

from hsr4hci.fitting import CircularGauss2D
from hsr4hci.general import crop_center
from hsr4hci.masking import get_circle_mask


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def crop_psf_template(
    psf_template: np.ndarray,
    psf_radius: Quantity,
    rescale_psf: bool = True,
) -> np.ndarray:
    """
    Take a raw unsaturated PSF template, and crop it to a circle of
    radius `psf_radius` around its center, which is determined by
    fitting a 2D Gaussian to the template.

    Args:
        psf_template: A numpy array containing the unsaturated PSF
            template which we use to create the planet signal.
        psf_radius: The radius (as an astropy.units.Quantity that
            can be converted to pixels) of the aperture to which the
            PSF template is cropped and masked.
        rescale_psf: Whether or not to rescale the PSF template to the
            value range (0, 1].

    Returns:
        The cropped and circularly masked PSF template as a numpy array.
    """

    # If desired, rescale the PSF template into the value range (0, 1]
    scale_factor = np.max(psf_template) if rescale_psf else 1
    psf_rescaled = np.copy(psf_template) / scale_factor

    # Clip the PSF template to strictly positive values
    epsilon = np.finfo(np.float64).eps
    psf_clipped = np.clip(a=psf_rescaled, a_min=epsilon, a_max=None)

    # Create a mask for the centering process
    mask = get_circle_mask(
        mask_size=(psf_clipped.shape[0], psf_clipped.shape[1]), radius=5
    )

    # Fit the center of the clipped PSF template
    psf_clipped_center = centroid_2dg(data=psf_clipped, mask=~mask)

    # Create a circular mask and multiply it with the clipped PSF. The
    # resulting masked PSF is automatically cropped to its bounding box.
    circular_aperture = CircularAperture(
        positions=psf_clipped_center, r=psf_radius.to('pixel').value
    )
    circular_mask = circular_aperture.to_mask(method='exact')
    psf_masked = circular_mask.multiply(psf_clipped)

    return np.asarray(psf_masked)


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
