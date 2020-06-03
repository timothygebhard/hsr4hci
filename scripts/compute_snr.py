"""
Take a FITS file as an input and compute the SNR for a given position.

This script essentially provides a convenient command-line interface
to the `hsr4hci.utils.evaluation.compute_optimized_snr()` function.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Optional

import argparse
import os
import time

import astropy.units as units
import numpy as np

from hsr4hci.utils.fits import read_fits
from hsr4hci.utils.evaluation import compute_optimized_snr
from hsr4hci.utils.units import set_units_for_instrument


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_arguments() -> argparse.Namespace:
    """
    Parse and return the command line arguments.

    Returns:
        An `argparse.Namespace` object containing the command line
        options that were passed to this script.
    """

    # Set up a parser
    parser = argparse.ArgumentParser()

    # Add required arguments
    parser.add_argument(
        '--file',
        metavar='PATH',
        type=str,
        help='Path to the FITS file which contains the input data (i.e., '
             'frames with planet signal estimates). [Required]',
        required=True
    )
    parser.add_argument(
        '--x',
        metavar='X',
        type=float,
        help='Initial guess for the x-position at which to compute the SNR. '
             '[Required]',
        required=True
    )
    parser.add_argument(
        '--y',
        metavar='Y',
        type=float,
        help='Initial guess for the y-position at which to compute the SNR. '
             '[Required]',
        required=True
    )
    parser.add_argument(
        '--frame-index',
        metavar='N',
        type=int,
        help='Index of the frame in case the input FITS file contains a 3D '
             'array. Default: 0.',
        default=0
    )
    parser.add_argument(
        '--aperture-radius',
        metavar='VALUE <UNIT>',
        nargs='*',
        help='Radius of the reference apertures to be used. If only a number '
             'is given, this is interpreted as pixels. If a tuple of the form '
             '<value unit> is given (e.g., "--aperture-radius 0.1 arcsec"), '
             'the astropy.units package is used to try and parse this based '
             'on the values of PIXSCALE and LAMBDA_OVER_D. '
             'Default: 1.77 (pixel).',
        default=['1.77']
    )
    parser.add_argument(
        '--ignore-neighbors',
        metavar='N',
        type=int,
        help='Number of reference apertures to the left and right of the '
             'target position that should be ignored to avoid biasing the '
             'noise estimate (e.g., due to self-subtraction wings). '
             'Default: 0.',
        default=0
    )
    parser.add_argument(
        '--target',
        metavar='TARGET',
        type=str,
        choices=('none', 'snr', 'fpf', 'signal', 'noise'),
        help='The quantity that should be optimized by varying the position '
             'slightly around the initial guess. If "none" is chosen, no such '
             'optimization is performed. Default: "snr".',
        default='snr'
    )
    parser.add_argument(
        '--method',
        metavar='METHOD',
        type=str,
        help='A string containing the optimization method to be used. This '
             'must either be "brute" for brute-force optimization via a grid '
             'search, or an optimization method supported by '
             'scipy.optimize.minimize() such as, for example, "nelder-mead". '
             'See the scipy docs for more details on this. Default: "brute".',
        default='brute'
    )
    parser.add_argument(
        '--grid-size',
        metavar='N',
        type=int,
        help='Size of the grid that is searched when using brute-force '
             'optimization. All other optimization methods ignore this '
             'parameter. Please note that the search time scales '
             '*quadratically* with this parameter! Default: 16.',
        default=16
    )
    parser.add_argument(
        '--max-distance',
        metavar='D',
        type=float,
        help='The maximum distance between the initial guess for the position '
             'and the final position found by the optimizer. Essentially, '
             'this parameter controls how far away from the initial guess we '
             'can get by optimizing the position. The value is to be given '
             'in units of pixels. Default: 1.0.',
        default=1.0
    )
    parser.add_argument(
        '--max-runtime',
        metavar='T',
        type=int,
        help='Maximum runtime for the optimization in seconds. If this value '
             'is exceeded, a TimeoutError is caused and the optimization is'
             'aborted. Default: 30 (seconds).',
        default=30
    )
    parser.add_argument(
        '--PIXSCALE',
        metavar='P',
        type=float,
        help='Pixel scale of the instrument that was used to obtain the input '
             'data, in units of arcseconds per pixel. Default: 0.0271 (this '
             'is the correct value for VLT/NACO).',
        default=0.0271
    )
    parser.add_argument(
        '--LAMBDA-OVER-D',
        metavar='L',
        type=float,
        help='Ratio of the wavelength and the size of the primary mirror, in '
             'units of arcseconds. Default: 0.096 (this is the correct value '
             'for L\' band data at the VLT).',
        default=0.096
    )

    # Parse the command line arguments and return the result
    return parser.parse_args()


def print_dictionary(
    dictionary: dict,
) -> None:
    """
    Print a dictionary in a clear, table-like way.

    Args:
        dictionary: A Python dictionary.
    """

    # Determine the length of the longest key
    max_key_length = max(len(_) for _ in dictionary.keys())

    # Loop over dictionary items and print them
    for key, value in dictionary.items():
        print(f'{key + ":":<{max_key_length}}', '\t', value)


# -----------------------------------------------------------------------------
# MAIN DEFINITIONS
# -----------------------------------------------------------------------------

def main() -> None:
    """
    Main code of the script. It is encapsulated as a function to make it
    available as an entry point in the setup.py of the package.
    """

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCOMPUTE SIGNAL-TO-NOISE RATIO FROM FITS FILE\n', flush=True)

    # -------------------------------------------------------------------------
    # Parse command line arguments
    # -------------------------------------------------------------------------

    # Get command line arguments
    args = get_arguments()

    # Get the PIXSCALE and LAMBDA_OVER_D first, because we need their values
    # to process the other command line arguments
    pixscale = units.Quantity(float(args.PIXSCALE), 'arcsec / pixel')
    lambda_over_d = units.Quantity(float(args.LAMBDA_OVER_D), 'arcsec')
    set_units_for_instrument(pixscale=pixscale, lambda_over_d=lambda_over_d)

    # Get path to FITS file and make sure it exists
    file_path = str(args.file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'{file_path} does not exist!')

    # Get the initial guess for the planet position
    position = (float(args.x), float(args.y))

    # Get the frame index
    frame_index = int(args.frame_index)

    # Get the aperture radius and cast its value to units of pixel
    aperture_radius = args.aperture_radius
    if len(aperture_radius) > 1:
        aperture_radius = \
            units.Quantity(float(aperture_radius[0]), str(aperture_radius[1]))
    else:
        aperture_radius = units.Quantity(float(aperture_radius[0]), 'pixel')

    # Get the number of neighboring apertures to ignore
    ignore_neighbors = int(args.ignore_neighbors)

    # Get the optimization target
    target: Optional[str] = str(args.target)
    if target == 'none':
        target = None

    # Get the optimization method
    method = str(args.method)

    # Get the grid size for brute-force optimization
    grid_size = int(args.grid_size)

    # Get the maximum distance for the optimizer
    max_distance = float(args.max_distance)

    # Get the maximum runtime for the optimizer
    max_runtime = int(args.max_runtime)

    # -------------------------------------------------------------------------
    # Print options that we have received
    # -------------------------------------------------------------------------

    options = {'PIXSCALE': pixscale.to_string(),
               'LAMBDA_OVER_D': lambda_over_d.to_string(),
               'file_path': file_path,
               'position': position,
               'frame_index': frame_index,
               'aperture_radius': aperture_radius.to_string(),
               'ignore_neighbors': ignore_neighbors,
               'target': target,
               'method': method,
               'grid_size': grid_size,
               'max_distance': max_distance,
               'max_runtime': max_runtime}

    print('Running with the following options:\n' + 80 * '-', flush=True)
    print_dictionary(options)
    print(80 * '-', flush=True)

    # -------------------------------------------------------------------------
    # Read in the FITS file and select the frame
    # -------------------------------------------------------------------------

    # Read the FITS file
    fits_array: np.ndarray = read_fits(file_path=file_path)

    # Check if we need to add an additional dimension
    if fits_array.ndim == 2:
        fits_array = fits_array[np.newaxis, ...]

    # Select the frame for which we want to compute the SNR
    frame = fits_array[frame_index]

    # -------------------------------------------------------------------------
    # Run the SNR computation and print the results
    # -------------------------------------------------------------------------

    results = \
        compute_optimized_snr(frame=frame,
                              position=position,
                              aperture_radius=aperture_radius.to('pix').value,
                              ignore_neighbors=ignore_neighbors,
                              target=target,
                              max_distance=max_distance,
                              method=method,
                              grid_size=grid_size,
                              time_limit=max_runtime)

    print('\nResults:\n' + 80 * '-')
    print_dictionary(results)
    print(80 * '-', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    main()
