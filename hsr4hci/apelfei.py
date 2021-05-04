"""
Functions which are (temporarily) ported from apelfei.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Optional, Tuple

import warnings

from astropy.modeling import models, fitting
from photutils import aperture_photometry, CircularAperture

import numpy as np

from hsr4hci.coordinates import get_center


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_flux(
    frame: np.ndarray,
    position: Tuple[float, float],
    mode: str = 'AS',
    aperture_radius: Optional[float] = None,
    search_radius: Optional[float] = None,
) -> Tuple[Tuple[float, float], float]:
    """
    This function estimates the flux at and / or around a given position
    in a frame.

    Args:
        frame: A 2D numpy array of shape `(width, height)` containing
            the data on which to run the aperture photometry.
        position: A tuple `(x, y)` specifying the position at which we
            estimate the flux.
        mode: Five different modes are supported:
            1.) "AS" (aperture sum): Compute the integrated flux over a
                circular aperture with radius `aperture_radius` at the
                given `position`. This is perhaps the most "intuitive"
                way to compute the flux.
            2.) "ASS" (aperture sum + search): Similar to "AS", except
                the `position` of the circular aperture is varied in a
                circular region with radius `search_radius` to find the
                position with the highest flux.
            3.) "P" (pixel): Compute or interpolate the value of a
                single pixel at the given `position`.
            4.) "F" (fit): Compute the flux by fitting a 2D Gaussian to
                the given position and returning its `amplitude`.
            5.) "FS" (fit + search): Similar to "F", except the position
                of the 2D Gaussian is also optimized within the given
                `search_radius`.
        aperture_radius: Required for modes "AS" and "ASS". Defines the
            radius of the circular aperture over which we integrate the
            flux.
        search_radius: Required for modes "ASS" and "FS". Defines the
            radius of the circular region within which we vary the
            position to optimize the flux.

    Returns:
        A tuple `(final_position, flux)`, where the `final_position` is
        a 2-tuple of floats.
    """

    # -------------------------------------------------------------------------
    # Modes based on apertures ('AS, 'ASS' and 'P')
    # -------------------------------------------------------------------------

    if mode in ('AS', 'ASS', 'P'):

        # In mode 'P', we use a circular aperture with a diameter of one pixel;
        # for the other modes, the user needs to specify an aperture_radius
        if mode == 'P':
            aperture_radius = 0.5
        elif mode in ('AS', 'ASS') and aperture_radius is None:
            raise ValueError('Modes "AS" and "ASS" need an aperture_radius!')

        # For search-based mode: Define a grid of positions at which we compute
        # the flux for the brute force optimization
        if mode == 'ASS':
            if search_radius is not None:
                offset = np.linspace(-search_radius, search_radius, 5)
                new_positions = (
                    np.array(
                        np.meshgrid(offset + position[0], offset + position[1])
                    )
                    .reshape(2, -1)
                    .T
                )
            else:
                raise ValueError('Mode "ASS" needs a search_radius!')
        else:
            new_positions = position

        # Set up an aperture (or a grid of apertures, for search-based mode)
        aperture = CircularAperture(positions=new_positions, r=aperture_radius)

        # Run aperture photometry, that is, compute the integrated flux for the
        # aperture (or each aperture on the grid)
        photometry_table = aperture_photometry(frame, aperture, method='exact')

        # Find the optimum, that is, the position with the highest flux, and
        # the corresponding flux. For modes that are not search-based, the
        # photometry table only contains a single row, which is automatically
        # the optimum.
        best_idx = np.argmax(photometry_table['aperture_sum'])
        best_aperture_sum = float(photometry_table['aperture_sum'][best_idx])
        best_position = (
            float(photometry_table['xcenter'][best_idx].value),
            float(photometry_table['ycenter'][best_idx].value),
        )

        return best_position, best_aperture_sum

    # -------------------------------------------------------------------------
    # Modes based on fitting a 2D Gaussian ('F' and 'FS')
    # -------------------------------------------------------------------------

    if mode in ('F', 'FS'):

        # Define the grid for the fit
        x = np.arange(frame.shape[0])
        y = np.arange(frame.shape[1])
        x, y = np.meshgrid(x, y)

        # Create a new Gaussian2D object
        gaussian_model = models.Gaussian2D(
            x_mean=position[0], y_mean=position[1]
        )

        # Enforce symmetry: tie standard deviation parameters to same value to
        # ensure that the resulting 2D Gaussian is always circular
        def tie_stddev(gaussian_model: Any) -> Any:
            return gaussian_model.y_stddev

        gaussian_model.x_stddev.tied = tie_stddev

        # Either fix the position, or define an area of admissible positions
        if mode == 'F':
            gaussian_model.x_mean.fixed = True
            gaussian_model.y_mean.fixed = True
        elif mode == 'FS' and search_radius is not None:
            gaussian_model.x_mean.min = position[0] - search_radius
            gaussian_model.x_mean.max = position[0] + search_radius
            gaussian_model.y_mean.min = position[1] - search_radius
            gaussian_model.y_mean.max = position[1] + search_radius

        # Fit the model to the data
        fit_p = fitting.LevMarLSQFitter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gaussian_model = fit_p(gaussian_model, x, y, frame)

        # Get the final position of the Gaussian after the fit
        final_position = (
            gaussian_model.x_mean.value,
            gaussian_model.y_mean.value,
        )

        # We cannot use the amplitude parameter of the Gaussian directly, as
        # it is not comparable with the values estimated in pixel mode (due to
        # the normalization of the Gaussian).
        # Therefore, we create a new frame that contains *only* the Gaussian
        # and perform "P" mode-like aperture photometry on this new frame.
        gaussian_frame = gaussian_model(x, y)
        aperture = CircularAperture(positions=position, r=0.5)
        photometry_table = aperture_photometry(
            gaussian_frame, aperture, method='exact'
        )
        flux = photometry_table['aperture_sum'][0]

        return final_position, flux

    # -------------------------------------------------------------------------
    # All other values for mode result in an error
    # -------------------------------------------------------------------------

    raise ValueError(f'Mode "{mode}" not supported!')


def get_stellar_flux(
    psf_template: np.ndarray,
    dit_stack: float,
    dit_psf_template: float,
    mode: str = 'FS',
    scaling_factor: float = 1.0,
    aperture_radius: Optional[float] = None,
    search_radius: float = 0.5,
) -> float:
    """
    This function takes the unsaturated PSF template and computes the
    flux of the star, normalized relative to the integration time of
    the stack.

    Args:
        psf_template: 2D numpy array with the unsaturated PSF template.
        dit_stack: Integration time of the frames in the stack.
        dit_psf_template: Integration time of the unsaturated PSF
            template.
        scaling_factor: A scaling factor to account for ND filters.
        mode: The mode which is used to estimate the flux of the star.
            5 different modes are supported. Mode "FS" recommended.
            See `get_aperture_flux()` for more details.
        aperture_radius: Needed for modes "AS" and "ASS". Gives the
            aperture radius of the circular aperture over which the
            flux is integrated.
        search_radius: Needed for modes "ASS" and "FS". Gives the search
            area which is considered to find the highest flux.

    Returns:
        The stellar flux, normalized relative to the DIT of the stack.
    """

    # Normalize the PSF template to account for the different detector
    # integration times (DIT) of the stack and the PSF template, as well as
    # the usage of a neutral density filter
    psf_normalized = (
        np.copy(psf_template) * dit_stack / dit_psf_template * scaling_factor
    )

    # Compute the flux at the center of the PSF template
    _, flux = get_flux(
        frame=psf_normalized,
        position=get_center(psf_normalized.shape),
        mode=mode,
        aperture_radius=aperture_radius,
        search_radius=search_radius,
    )

    return flux
