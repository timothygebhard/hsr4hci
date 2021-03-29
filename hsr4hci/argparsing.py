"""
Utility functions for parsing command line arguments.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import os


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_base_directory(required: bool = True) -> Path:
    """
    Parse command line arguments given to the script and return value
    of the --base-directory flag as a string.

    Args:
        required: Whether or not --base-directory is a required flag;
            that is, if it not being passed to the script calling this
            function will raise an Exception.

    Returns:
        The value of the --base-directory flag as a string.
    """

    # Set up the parser and its arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base-directory',
        type=str,
        metavar='PATH',
        required=required,
        help=(
            'Path to the base directory containing the config.json which '
            'specifies the data set and how it should be pre-processed.'
        ),
    )

    # Parse the arguments and return the base_directory
    args = parser.parse_args()
    return Path(os.path.realpath(args.base_directory))
