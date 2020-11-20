"""
Utility functions for working with point spread functions.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Optional

from astropy.convolution import AiryDisk2DKernel
from astropy.units import Quantity
from photutils import centroid_2dg, CircularAperture

import numpy as np

from hsr4hci.utils.fitting import fit_2d_function, moffat_2d
from hsr4hci.utils.general import crop_center
from hsr4hci.utils.masking import get_circle_mask


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
    psf_rescaled = psf_template / scale_factor

    # Clip the PSF template to strictly positive values
    epsilon = np.finfo(np.float64).eps
    psf_clipped = np.clip(a=psf_rescaled, a_min=epsilon, a_max=None)

    # Create a mask for the centering process
    mask = get_circle_mask(mask_size=psf_clipped.shape, radius=5)

    # Fit the center of the clipped PSF template
    psf_clipped_center = centroid_2dg(data=psf_clipped, mask=~mask)

    # Create a circular mask and multiply it with the clipped PSF. The
    # resulting masked PSF is automatically cropped to its bounding box.
    circular_aperture = CircularAperture(
        positions=psf_clipped_center, r=psf_radius.to('pixel').value
    )
    circular_mask = circular_aperture.to_mask(method='exact')
    psf_masked = circular_mask.multiply(psf_clipped)

    return psf_masked


def get_psf_diameter(
    psf_template: np.ndarray,
    pixscale: Optional[float] = None,
    lambda_over_d: Optional[float] = None,
) -> float:
    """
    Fit a 2D Moffat function to the given PSF template to estimate
    the diameter of the central "blob" in pixels.

    The diameter is computed at the arithmetic mean of the FWHM in
    x and y direction, as returned by the fit.

    Args:
        psf_template: A 2D numpy array containing the raw, unsaturated
            PSF template.
        pixscale:
        lambda_over_d:

    Returns:
        The diameter of the PSF template in pixels.
    """

    # Case 1: We have been provided a suitable PSF template and can determine
    # the size by fitting the PSF with a Moffat function
    if psf_template.shape[0] >= 33 and psf_template.shape[1] >= 33:

        # Crop PSF template: too large templates (which are mostly zeros) can
        # cause problems when fitting them with a 2D Moffat function
        psf_template = crop_center(psf_template, (33, 33))

        # Define shortcuts
        psf_center_x = float(psf_template.shape[0] / 2)
        psf_center_y = float(psf_template.shape[1] / 2)

        # Define initial guess for parameters
        p0 = (psf_center_x, psf_center_y, 1, 1, 1, 0, 0, 1)

        # Fit the PSF template with a 2D Moffat function to get the FWHMs
        params = fit_2d_function(frame=psf_template, function=moffat_2d, p0=p0)

        # Compute the PSF diameter as the mean of the two FWHM values
        fwhm_x, fwhm_y = params[2:4]
        psf_diameter = float(0.5 * (fwhm_x + fwhm_y))

    # Case 2: We do not have PSF template, but the PIXSCALE and LAMBDA_OVER_D
    elif (pixscale is not None) and (lambda_over_d is not None):

        # In this case, we can approximately compute the expected PSF size.
        # The 1.144 is a magic number to get closer to the empirical estimate
        # from data sets where a PSF template is available.
        psf_diameter = lambda_over_d / pixscale * 1.144

    # Case 3: In all other scenarios, we raise an error
    else:
        raise RuntimeError('Could not determine PSF diameter')

    return psf_diameter


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
    return AiryDisk2DKernel(
        radius=1.383 * (lambda_over_d / pixscale).to('pixel').value,
    ).array
