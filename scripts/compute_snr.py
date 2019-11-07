"""
Take a FITS file as input and compute the SNR for a given position.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse

from hsr4hci.utils.fits import read_fits
from hsr4hci.utils.evaluation import compute_figures_of_merit


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

    # Allow us to split the arguments into required and optional ones
    # noinspection PyProtectedMember
    # pylint: disable=protected-access
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')

    # Add required arguments
    required.add_argument('--file',
                          metavar='PATH',
                          type=str,
                          help='Path to the FITS file containing the data',
                          required=True)
    required.add_argument('--x',
                          metavar='X',
                          type=float,
                          help='x-position for which to compute the SNR',
                          required=True)
    required.add_argument('--y',
                          metavar='Y',
                          type=float,
                          help='y-position for which to compute the SNR',
                          required=True)

    # Add optional arguments
    optional.add_argument('--aperture-size',
                          type=float,
                          metavar='X',
                          default=1.77,
                          help='Aperture radius in pixels (default: 1.77)')
    optional.add_argument('--optimize',
                          type=str,
                          default='snr',
                          metavar='STR',
                          choices=['none', 'signal', 'noise_level', 'snr',
                                   'fpf'],
                          help='Quantity to optimize, must be in ["none", '
                               '"signal", "noise_level", "snr", "fpf"] '
                               '(default: "snr")')

    # Add mutually exclusive flags --ignore-neighbors / --include-neighbors
    ignore_neighbors_parser = \
        optional.add_mutually_exclusive_group(required=False)
    ignore_neighbors_parser.add_argument('--ignore-neighbors',
                                         dest='ignore_neighbors',
                                         action='store_true',
                                         help='Ignore neighboring apertures '
                                              '(default)')
    ignore_neighbors_parser.add_argument('--include-neighbors',
                                         dest='ignore_neighbors',
                                         action='store_false',
                                         help='Include neighboring apertures')
    optional.set_defaults(ignore_neighbors=True)

    # Make sure optional arguments also show up when using --help
    # noinspection PyProtectedMember
    # pylint: disable=protected-access
    parser._action_groups.append(optional)

    # Parse the command line arguments and return the result
    return parser.parse_args()


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    print('\nCOMPUTE FIGURES OF MERIT FROM FITS FILE\n', flush=True)

    # Get command line arguments
    args = get_arguments()

    # Define shortcuts to arguments (and typecast)
    file_path = args.file
    position = (float(args.x), float(args.y))
    aperture_size = float(args.aperture_size)
    ignore_neighbors = args.ignore_neighbors
    optimize = args.optimize if args.optimize != 'none' else None

    # Load the FITS file
    frame = read_fits(file_path=file_path)

    # Compute figures of merit
    figures_of_merit = \
        compute_figures_of_merit(frame=frame,
                                 position=position,
                                 aperture_size=aperture_size,
                                 ignore_neighbors=ignore_neighbors,
                                 optimize=optimize)

    # Print results
    for key, value in figures_of_merit.items():
        print(f'{key:<20}{value}')
    print()
