"""
Utilities for reading in config files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import json
import os

from hsr4hci.utils.units import convert_to_quantity, set_units_for_instrument


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def load_config(config_file_path: str) -> dict:
    """
    Load and augment an experiment configuration.

    Args:
        config_file_path: Path to the JSON file containing the
            configuration to be loaded.

    Returns:
        A dictionary containing the augmented configuration.
    """

    # -------------------------------------------------------------------------
    # Load configuration from JSON file
    # -------------------------------------------------------------------------

    # Build the full path to the config file and check if it exists
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f'{config_file_path} does not exist!')

    # Load the config file into a dict
    with open(config_file_path, 'r') as json_file:
        config = json.load(json_file)

    # -------------------------------------------------------------------------
    # Augment configuration (i.e., add implicitly defined variables)
    # -------------------------------------------------------------------------

    # Add the path to the experiments folder to the config dict
    config['experiment_dir'] = os.path.dirname(config_file_path) or '.'

    # Replace dummy data directory with machine-specific data directory
    data_dir = get_data_dir()
    config['dataset']['file_path'] = \
        config['dataset']['file_path'].replace('DATA_DIR', data_dir)

    # -------------------------------------------------------------------------
    # Convert values into astropy.units.Quantity objects
    # -------------------------------------------------------------------------

    # First, convert pixscale and lambda_over_d to astropy.units.Quantity
    config = convert_to_quantity(config, ('dataset', 'pixscale'))
    config = convert_to_quantity(config, ('dataset', 'lambda_over_d'))

    # Use this to set up the instrument-specific conversion factors. We need
    # this here to that we can parse "lambda_over_d" as a unit in the config.
    set_units_for_instrument(pixscale=config['dataset']['pixscale'],
                             lambda_over_d=config['dataset']['lambda_over_d'])

    # Convert the remaining entries of the config to astropy.units.Quantity
    for key_tuple in [('roi_mask', 'inner_radius'),
                      ('roi_mask', 'outer_radius'),
                      ('selection_mask', 'annulus_width'),
                      ('selection_mask', 'radius_position'),
                      ('selection_mask', 'radius_mirror_position'),
                      ('selection_mask', 'minimum_distance')]:
        config = convert_to_quantity(config, key_tuple)

    return config


def get_data_dir() -> str:
    """
    Get the path to the data directory from an environment variable.

    Returns:
        Path to the data directory.
    """

    # Check if the HSR4HCI_DATA_DIR environment variable is set
    if 'HSR4HCI_DATA_DIR' in os.environ.keys():
        data_dir = os.environ['HSR4HCI_DATA_DIR']
    else:
        raise RuntimeError('Environment variable HSR4HCI_DATA_DIR not set!')

    # Check if the value it contains is a valid directory
    if not os.path.isdir(data_dir):
        raise RuntimeError('Value of HSR4HCI_DATA_DIR is not a directory!')

    return data_dir
