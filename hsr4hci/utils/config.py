"""
Utilities for reading in config files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Dict

import json
import os


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def load_config(config_file_path: str) -> Dict[str, Any]:
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
        config: Dict[str, Any] = json.load(json_file)

    # -------------------------------------------------------------------------
    # Augment configuration (i.e., add implicitly defined variables)
    # -------------------------------------------------------------------------

    # Add the path to the experiments folder to the config dict
    config['experiment_dir'] = os.path.dirname(config_file_path) or '.'

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
