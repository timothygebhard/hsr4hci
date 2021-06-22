"""
Functions for measuring fluxes.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, List, Optional, Tuple

from astropy.modeling import models, fitting
from astropy.units import Quantity
from photutils import aperture_photometry, CircularAperture

import numpy as np

from hsr4hci.coordinates import get_center, polar2cartesian


# -----------------------------------------------------------------------------
# AUXILIARY FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def _get_flux__as(
    frame: np.ndarray,
    position: Tuple[float, float],
    aperture_radius: Quantity,
) -> Tuple[Tuple[float, float], float]:
    """
    Auxiliary function to measure the flux using the "AS" mode.
    See `get_flux()` for more details.
    """

    # Set up an aperture of the given size
    aperture = CircularAperture(
        positions=position, r=aperture_radius.to('pixel').value
    )

    # Compute the integrated flux for the aperture; get flux
    photometry_table = aperture_photometry(frame, aperture)
    flux = float(photometry_table['aperture_sum'][0])

    return position, flux


def _get_flux__ass(
    frame: np.ndarray,
    position: Tuple[float, float],
    aperture_radius: Quantity,
    search_radius: Quantity,
) -> Tuple[Tuple[float, float], float]:
    """
    Auxiliary function to measure the flux using the "ASS" mode.
    See `get_flux()` for more details.
    """

    # Construct a 2D grid of positions around the target position at which we
    # will compute the flux in order to maximize it
    offset = float(search_radius.to('pixel').value)
    offsets = np.linspace(-offset, offset, 21)
    positions_grid = np.meshgrid(offsets + position[0], offsets + position[1])
    positions_grid = np.asarray(positions_grid).reshape(2, -1).T

    # Set up a 2D grid of apertures of the given size
    aperture = CircularAperture(
        positions=positions_grid, r=aperture_radius.to('pixel').value
    )

    # Compute the integrated flux for each aperture in the 2D grid
    photometry_table = aperture_photometry(frame, aperture)

    # Find the optimum, that is, the position with the highest flux, as well
    # as the corresponding flux
    best_idx = np.argmax(photometry_table['aperture_sum'])
    best_flux = float(photometry_table['aperture_sum'][best_idx])
    best_position = (
        float(photometry_table['xcenter'][best_idx].value),
        float(photometry_table['ycenter'][best_idx].value),
    )

    return best_position, best_flux


def _get_flux__p(
    frame: np.ndarray,
    position: Tuple[float, float],
) -> Tuple[Tuple[float, float], float]:
    """
    Auxiliary function to measure the flux using the "P" mode.
    See `get_flux()` for more details.
    """

    # Set up an aperture with a radius of 0.5 pixels
    aperture = CircularAperture(positions=position, r=0.5)

    # Compute the integrated flux for the aperture; get flux
    photometry_table = aperture_photometry(frame, aperture)
    flux = float(photometry_table['aperture_sum'][0])

    return position, flux


def _get_flux__f(
    frame: np.ndarray,
    position: Tuple[float, float],
) -> Tuple[Tuple[float, float], float]:
    """
    Auxiliary function to measure the flux using the "F" mode.
    See `get_flux()` for more details.
    """

    # Define the grid for the fit
    x = np.arange(frame.shape[0])
    y = np.arange(frame.shape[1])
    x, y = np.meshgrid(x, y)

    # Create a new Gaussian2D object
    gaussian_model = models.Gaussian2D(x_mean=position[0], y_mean=position[1])

    # Define auxiliary function for tieing the standard deviations
    def tie_stddev(gaussian_model: Any) -> Any:
        return gaussian_model.y_stddev

    # Enforce symmetry: tie standard deviation parameters to same value to
    # ensure that the resulting 2D Gaussian is always circular
    gaussian_model.x_stddev.tied = tie_stddev

    # Fix the position (= mean) of the 2D Gaussian
    gaussian_model.x_mean.fixed = True
    gaussian_model.y_mean.fixed = True

    # Fit the model to the data
    fit_p = fitting.LevMarLSQFitter()
    gaussian_model = fit_p(gaussian_model, x, y, np.nan_to_num(frame))

    # Get the final position of the Gaussian after the fit
    final_position = (gaussian_model.x_mean.value, gaussian_model.y_mean.value)

    # We cannot use the amplitude parameter of the Gaussian directly, as
    # it is not comparable with the values estimated in pixel mode (due to
    # the normalization of the Gaussian). Therefore, we create a new frame
    # that contains *only* the Gaussian run "P"-mode style photometry.
    return _get_flux__p(frame=gaussian_model(x, y), position=final_position)


def _get_flux__fs(
    frame: np.ndarray,
    position: Tuple[float, float],
    search_radius: Quantity,
) -> Tuple[Tuple[float, float], float]:
    """
    Auxiliary function to measure the flux using the "FS" mode.
    See `get_flux()` for more details.
    """

    # Define the grid for the fit
    x = np.arange(frame.shape[0])
    y = np.arange(frame.shape[1])
    x, y = np.meshgrid(x, y)

    # Create a new Gaussian2D object
    gaussian_model = models.Gaussian2D(x_mean=position[0], y_mean=position[1])

    # Define auxiliary function for tieing the standard deviations
    def tie_stddev(gaussian_model: Any) -> Any:
        return gaussian_model.y_stddev

    # Enforce symmetry: tie standard deviation parameters to same value to
    # ensure that the resulting 2D Gaussian is always circular
    gaussian_model.x_stddev.tied = tie_stddev

    # Define "search area" by setting minimum and maximum values for the mean
    # of the Gaussian (i.e., the position)
    gaussian_model.x_mean.min = position[0] - search_radius.to('pixel').value
    gaussian_model.x_mean.max = position[0] + search_radius.to('pixel').value
    gaussian_model.y_mean.min = position[1] - search_radius.to('pixel').value
    gaussian_model.y_mean.max = position[1] + search_radius.to('pixel').value

    # Fit the model to the data
    fit_p = fitting.LevMarLSQFitter()
    gaussian_model = fit_p(gaussian_model, x, y, np.nan_to_num(frame))

    # Get the final position of the Gaussian after the fit
    final_position = (gaussian_model.x_mean.value, gaussian_model.y_mean.value)

    # We cannot use the amplitude parameter of the Gaussian directly, as
    # it is not comparable with the values estimated in pixel mode (due to
    # the normalization of the Gaussian). Therefore, we create a new frame
    # that contains *only* the Gaussian run "P"-mode style photometry.
    return _get_flux__p(frame=gaussian_model(x, y), position=final_position)


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_flux(
    frame: np.ndarray,
    position: Tuple[float, float],
    mode: str = 'AS',
    aperture_radius: Optional[Quantity] = None,
    search_radius: Optional[Quantity] = None,
) -> Tuple[Tuple[float, float], float]:
    """
    This function estimates the flux at or around a given position in
    a frame. There are five different modes for how to do this:

        1. "AS" (aperture sum):
            Compute the integrated flux over a circular aperture with
            radius `aperture_radius` at the given `position`. This is
            perhaps the most "intuitive" way to compute the flux.
        2. "ASS" (aperture sum + search):
            Similar to "AS", except the `position` of the circular
            aperture is varied in a circular region with radius
            `search_radius` to find the position with the highest flux.
        3. "P" (pixel):
            Compute or interpolate the value of a single pixel at the
            given `position`.
        4. "F" (fit):
            Compute the flux by fitting a 2D Gaussian to the given
            position and returning its `amplitude`.
        5. "FS" (fit + search):
            Similar to "F", except the position of the 2D Gaussian is
            also optimized within the given `search_radius`.

    Args:
        frame: A 2D numpy array of shape `(width, height)` containing
            the data on which to run the aperture photometry.
        position: A tuple `(x, y)` specifying the position at which to
            estimate the flux.
        mode: See above.
        aperture_radius: Required for modes "AS" and "ASS". Defines the
            radius of the circular aperture over the flux is integrated.
        search_radius: Required for modes "ASS" and "FS". Defines the
            size of the region within which we vary the position to find
            the "optimal" (= highest) flux.

    Returns:
        A tuple `(final_position, flux)`, where the `final_position` is
        a 2-tuple of floats (i.e., the Cartesian position using the
        photutils coordinate convention).
    """

    # Compute the flux according to the selected mode
    if mode == 'AS':
        return _get_flux__as(frame, position, aperture_radius)
    if mode == 'ASS':
        return _get_flux__ass(frame, position, aperture_radius, search_radius)
    if mode == 'P':
        return _get_flux__p(frame, position)
    if mode == 'F':
        return _get_flux__f(frame, position)
    if mode == 'FS':
        return _get_flux__fs(frame, position, search_radius)

    # All other values for mode result in an error
    raise ValueError(f'Mode "{mode}" not supported!')


def get_stellar_flux(
    psf_template: np.ndarray,
    dit_stack: float,
    dit_psf_template: float,
    mode: str = 'FS',
    scaling_factor: float = 1.0,
    aperture_radius: Optional[Quantity] = None,
    search_radius: Optional[Quantity] = Quantity(1, 'pixel'),
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
        mode: See `get_aperture_flux()` for more details. For the
            stellar flux, mode "FS" is recommended.
        aperture_radius: See `get_aperture_flux()` for more details.
        search_radius: See `get_aperture_flux()` for more details.

    Returns:
        The stellar flux, normalized relative to the DIT of the stack.
    """

    # Normalize the PSF template to account for the different detector
    # integration times (DIT) of the stack and the PSF template, as well
    # as for the usage of a neutral density filter
    psf_normalized = (
        np.copy(psf_template) * dit_stack / dit_psf_template / scaling_factor
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


def get_fluxes_for_polar_positions(
    polar_positions: List[Tuple[Quantity, Quantity]],
    frame: np.ndarray,
    mode: str = 'AS',
    aperture_radius: Optional[Quantity] = None,
    search_radius: Optional[Quantity] = None,
) -> List[float]:
    """
    Auxiliary function for applying to `get_flux()` to a list of
    positions that are given in ("astronomical") polar coordinates.

    Args:
        polar_positions: A list of positions in polar coordinates, that
            is, every position is a tuple `(separation, angle)`, where
            for the angle, 0 degrees is "up", not "right".
        frame: The frame / image on which to perform the photometry.
        mode: The `mode` parameter for `get_flux()`; see there for
            more details.
        aperture_radius: The `aperture_radius` parameter for
            `get_flux()`; see there for more details.
        search_radius: The `search_radius` parameter for
            `get_flux()`; see there for more details.

    Returns:
        A list of the fluxes for each given polar position.
    """

    # Determine frame size
    frame_size = (frame.shape[0], frame.shape[1])

    # Loop over polar positions and measure the flux at each one
    fluxes = []
    for polar_position in polar_positions:

        # Convert polar position to Cartesian one
        cartesian_position = polar2cartesian(
            separation=polar_position[0],
            angle=polar_position[1],
            frame_size=frame_size,
        )

        # Measure and store the flux for this position
        _, flux = get_flux(
            frame=np.nan_to_num(frame),
            position=cartesian_position,
            mode=mode,
            aperture_radius=aperture_radius,
            search_radius=search_radius,
        )
        fluxes.append(flux)

    return fluxes
