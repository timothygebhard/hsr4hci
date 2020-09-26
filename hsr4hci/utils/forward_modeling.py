"""
Utility functions for forward modeling (necessary for toy data sets!).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from cmath import polar
from typing import Dict, Tuple

from astropy.convolution import AiryDisk2DKernel
from astropy.units import Quantity

import numpy as np

from hsr4hci.utils.general import add_array_with_interpolation, rotate_position
from hsr4hci.utils.psf import crop_psf_template


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_signal_stack(
    position: Tuple[float, float],
    frame_size: Tuple[int, int],
    parang: np.ndarray,
    psf_cropped: np.ndarray,
) -> Tuple[np.ndarray, list]:
    """
    Compute the forward model: Assume that at time t=0, the planet is
    at the given `position. The apparent motion of the sky (and thus `
    the planet) is given by the parallactic angles in `parang`. This
    function now takes the provided cropped and masked PSF. Then, an
    empty stack of size `(len(parang), *frame_size)` is created, and
    for each time t, the cropped PSF template is added into the stack
    at the correct position according to parang[t].

    Args:
        position: A tuple containing the initial planet position (i.e.,
            we compute the forward model under the assumption that the
            planet is at this position for t=0).
        frame_size: A tuple with the spatial size `(x_size, y_size)`
            of the forward model stack to be created.
        parang: A numpy array containing the parallactic angle for
            every frame.
        psf_cropped: A numpy array containing the cropped and masked
            PSF template, as it is returned by `crop_psf_template()`.

    Returns:
        A tuple consisting of a numpy array containing the forward model
        (i.e., a stack of frames where every frame contains only the
        "planet signal" at the correct position), computed under the
        given assumptions, as well as a list of tuples (x, y) containing
        the position of the planet in the forward model for every frame.
    """

    # Define some shortcuts
    n_frames = len(parang)
    frame_width, frame_height = frame_size
    frame_center = (frame_width / 2, frame_height / 2)

    # Compute relative rotation angles (i.e., the rotation relative to t=0)
    rotation_angles = parang - parang[0]

    # Compute polar representation of initial position
    r, phi = polar(
        complex(position[1] - frame_center[1], position[0] - frame_center[0])
    )

    # Initialize empty signal stack and list of planet positions
    signal_stack = np.zeros((n_frames, frame_width, frame_height))
    planet_positions = list()

    # Loop over all frames and add the cropped and masked PSF template at
    # the position that corresponds to the respective parallactic angle
    for i in range(n_frames):

        # Compute injection position for the current frame: Rotate the initial
        # position by the correct angle (using the exponential representation
        # of 2D vectors) and convert the resulting position back to Cartesian
        # coordinates. Store this value to a list that we will return along
        # with the signal stack we have constructed.
        theta = np.deg2rad(rotation_angles[i])
        new_complex_position = r * np.exp(1j * (phi - theta))
        injection_position = (
            np.imag(new_complex_position) + frame_center[0],
            np.real(new_complex_position) + frame_center[1],
        )
        planet_positions.append(injection_position)

        # Add cropped and masked PSF template at the injection_position.
        # The add_array_with_interpolation() function is able to automatically
        # deal with cases where the injection_position is a tuple of floats,
        # or the PSF template exceeds the bounds of the signal stack.
        signal_stack[i] = add_array_with_interpolation(
            array_large=signal_stack[i],
            array_small=psf_cropped,
            position=injection_position,
        )

    return signal_stack, planet_positions


def get_planet_paths(
    stack_shape: Tuple[int, int, int],
    parang: np.ndarray,
    psf_template: np.ndarray,
    pixscale: Quantity,
    lambda_over_d: Quantity,
    planet_config: Dict[str, dict],
    threshold: float = 5e-1,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Auxiliary function for computing the paths of the planets in the
    data set (as a list of tuples indicating positions), as well as for
    creating binary masks that indicate which spatial pixels in a data
    set contain planet signal at some point in time.

    Args:
        stack_shape: A tuple `(n_frames, x_size, y_size)` giving the
            shape of the stack of the data set.
        parang: A 1D numpy array of shape `(n_frames, )` containing the
            parallactic angles.
        psf_template: A 2D numpy array containing the raw, unsaturated
            PSF template. If no such template is available, you can use
            a numpy array of size `(0, 0)` to automatically create a
            fake PSF template (using a 2D Airy function) to get an
            approximate solution.
        pixscale: An `astropy.units.Quantity` in units of arc seconds
            per pixel specifying the pixel scale of the instrument.
        lambda_over_d: An `astropy.units.Quantity` in units of arc
            seconds specifying the ratio between the filter wavelength
            lambda and the diameter of the telescope's primary mirror.
        planet_config: A dictionary containing information about the
            planets in the data set. Each key (usually the letter that
            indicates the name of the planet, e.g., "b") must map onto
            another dictionary, which contains a key named "position",
            which specifies the position of the planet in the *final*,
            derotated signal estimate (e.g., the PCA estimate).
        threshold: The threshold value for creating the binary mask;
            basically the minimum planet signal to be contained in a
            pixel (at some point) to add this pixel to the mask.

    Returns:
        A 2D numpy array of shape `(x_size, y_size)` in which every
        pixel that at some point in time contains planet signal is True
        and all other pixels are False.
    """

    # Define shortcuts
    n_frames, x_size, y_size = stack_shape
    center = (x_size / 2, y_size / 2)

    # In case there is no PSF template present, we need to create a fake
    # one using an Airy kernel of the appropriate size
    if psf_template.shape == (0, 0):

        # Create a 2D Airy disk of the correct size as a numpy array.
        # The factor of 1.383 is a "magic" number that was determined by
        # comparing real PSFs (for which the PIXSCALE and LAMBDA_OVER_D were
        # known) against the output of the AiryDisk2DKernel() function, and
        # adjusting the radius of the latter by a factor to minimize the
        # difference between the real and the fake PSF.
        psf_template = AiryDisk2DKernel(
            radius=1.383 * (lambda_over_d / pixscale).to('pixel').value,
            x_size=x_size,
            y_size=y_size,
        ).array

    # Crop the PSF to a 1 lambda_over_d region here (because we ignore the
    # secondary maxima for the planet path mask)
    psf_cropped = crop_psf_template(
        psf_template=psf_template,
        psf_radius=lambda_over_d,
    )

    # Instantiate an empty stack-like variable from which we will generate the
    # mask, and a dictionary which will hold the planet positions
    stack = np.zeros(stack_shape)
    all_planet_positions: Dict[str, np.ndarray] = dict()

    # Loop over the different planets for the data set
    for key, values in planet_config.items():

        # Get final position of the planet
        final_position = values['position'][::-1]

        # Compute the starting position of the planet
        starting_position = rotate_position(
            position=final_position,
            center=center,
            angle=parang[0],
        )

        # Compute a forward model for this planet
        planet_model, planet_positions = get_signal_stack(
            position=starting_position,
            frame_size=(x_size, y_size),
            parang=parang,
            psf_cropped=psf_cropped,
        )

        # Convert the planet positions to a 2D array and store them
        planet_positions = np.array(planet_positions)
        all_planet_positions[key] = planet_positions

        # Add this to the existing stack
        stack += planet_model

    # Compute the mask of the planet paths by taking the maximum along the
    # temporal dimension and thresholding the result
    planet_paths_mask = np.max(stack, axis=0) > threshold

    return planet_paths_mask, all_planet_positions
