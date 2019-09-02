"""
Utility methods for forward modeling.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np

from astropy.nddata import Cutout2D
from astropy.nddata.utils import add_array
from cmath import polar
from photutils import centroid_2dg, CircularAperture
from scipy import ndimage

from typing import Tuple, Union


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def add_array_with_interpolation(array_large: np.ndarray,
                                 array_small: np.ndarray,
                                 position: Tuple[Union[int, float],
                                                 Union[int, float]]):
    """
    An extension of astropy.nddata.utils.add_array to add a smaller
    array at a given position in a larger array. In this version, the
    position may also be a float, in which case bilinear interpolation
    is used when adding array_small into array_large.

    Args:
        array_large: Large array, into which array_small is added into.
        array_small: Small array, which is added into array_large.
        position: The target position of the small array’s center, with
            respect to the large array. Coordinates should be in the
            same order as the array shape, but can also be floats.

    Returns:
        The new array, constructed as the the sum of `array_large`
        and `array_small`.
    """

    # Create an empty with the same size as array_larger and add the
    # small array at the approximately correct position
    dummy = np.zeros_like(array_large)
    dummy = add_array(dummy, array_small, position)

    # Compute the subpixel offset and use scipy.ndimage.shift to shift the
    # array to the exact position, using bilinear interpolation
    offset = (position[0] % 1, position[1] % 1)
    dummy = ndimage.shift(dummy, offset, order=1)

    return array_large + dummy


def get_signal_stack(position: Tuple[int, int],
                     frame_size: Tuple[int, int],
                     parang: np.ndarray,
                     psf_template: np.ndarray,
                     psf_radius: float,
                     rescale_psf: bool = True,
                     pixscale: float = 0.0271,
                     lambda_over_d: float = 0.1) -> np.ndarray:
    """
    Compute the forward model: Assume that at time t=0, the planet is
    at the given `position. The apparent motion of the sky (and thus `
    the planet) is given by the parallactic angles in `parang`. This
    function now takes the provided `psf_template` and first crops it
    to a desired size (given by `psf_radius`). Then, an empty stack of
    size `(len(parang), *frame_size)` is created, and for each time t,
    the cropped PSF template is added into the stack at the correct
    position according to parang[t].

    Args:
        position: A tuple containing the initial planet position (i.e.,
            we compute the forward model under the assumption that the
            planet is at this position for t=0).
        frame_size: A tuple with the spatial size `(x_size, y_size)`
            of the forward model stack to be created.
        parang: A numpy array containing the parallactic angle for
            every frame.
        psf_template: A numpy array containing the unsaturated PSF
            template which we use to create the planet signal.
        psf_radius: The radius (in units of lambda over D) of the
            aperture to which the PSF template is cropped and masked.
        rescale_psf: Whether or not to rescale the PSF template to the
            value range (0, 1].
        pixscale: The resolution of the data, that is, the conversion
            factor between pixels and arcseconds. For VLT/NACO data,
            this value is usually 0.0271 arcseconds / pixel.
        lambda_over_d: lambda / D in arcseconds. For VLT/NACO (D=8.2m)
            data in the L band (lambda=3800nm), this value is ~0.1".

    Returns:
        A numpy array containing the forward model (i.e., a stack of
        frames where every frame contains only the "planet signal" at
        the correct position), computed under the given assumptions.
    """

    # -------------------------------------------------------------------------
    # Crop PSF template to a circular disk of the given radius
    # -------------------------------------------------------------------------

    # Convert psf_radius from units of lambda over D to pixel
    psf_radius_pixel = int(psf_radius * (lambda_over_d / pixscale))

    # If desired, rescale the PSF template into the value range (0, 1]
    scale_factor = np.max(psf_template) if rescale_psf else 1
    psf_rescaled = psf_template / scale_factor

    # Clip the PSF template to strictly positive values
    epsilon = np.finfo(np.float32).eps
    psf_clipped = np.clip(a=psf_rescaled, a_min=epsilon, a_max=None)

    # Fit the center of the clipped PSF template
    psf_clipped_center = centroid_2dg(psf_clipped)

    # Crop the PSF template around the center we have computed
    crop_size = (2 * psf_radius_pixel + 1, 2 * psf_radius_pixel + 1)
    psf_cropped = Cutout2D(data=psf_clipped,
                           position=psf_clipped_center,
                           size=crop_size).data

    # Create a circular mask and multiply it with the cropped PSF
    circular_aperture = \
        CircularAperture(positions=(psf_radius_pixel, psf_radius_pixel),
                         r=psf_radius_pixel)
    circular_mask = circular_aperture.to_mask(method='exact')
    psf_masked = circular_mask.multiply(psf_cropped)

    # -------------------------------------------------------------------------
    # Create the signal stack by injecting the PSF (= forward modeling)
    # -------------------------------------------------------------------------

    # Define some shortcuts
    n_frames = len(parang)
    frame_width, frame_height = frame_size
    frame_center = (frame_width / 2, frame_height / 2)

    # Compute relative rotation angles (i.e., the rotation relative to t=0)
    rotation_angles = parang - parang[0]

    # Compute polar representation of initial position
    r, phi = polar(complex(position[1] - frame_center[1],
                           position[0] - frame_center[0]))

    # Initialize empty signal stack
    signal_stack = np.zeros((n_frames, frame_width, frame_height))

    # Loop over all frames and add the cropped and masked PSF template at
    # the position that corresponds to the respective parallactic angle
    for i in range(n_frames):

        # Compute injection position for this frame:
        # Rotate the initial  position by the respective angle (using
        # the exponential representation of 2D vectors) and convert the
        # resulting position back to Cartesian coordinates.
        theta = np.deg2rad(rotation_angles[i])
        new_complex_position = r * np.exp(1j * (phi + theta))
        injection_position = (np.imag(new_complex_position) + frame_center[0],
                              np.real(new_complex_position) + frame_center[1])

        # Add cropped and masked PSF template at the injection_position.
        # The add_array_with_interpolation() function is able to automatically
        # deal with cases where the injection_position is a tuple of floats,
        # or the PSF template exceeds the bounds of the signal stack.
        signal_stack[i] = \
            add_array_with_interpolation(array_large=signal_stack[i],
                                         array_small=psf_masked,
                                         position=injection_position)

    return signal_stack


def get_collection_region_mask(signal_stack: np.ndarray) -> np.ndarray:
    return np.sum(signal_stack, axis=0) > 0


def get_collection_region_pixels(collection_region_mask: np.ndarray) -> list:
    return list(zip(*np.where(collection_region_mask)))
