"""
Utilities for reading in config files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, Union

import json

import hsr4hci


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def load_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and augment an experiment configuration.

    Args:
        file_path: Path to the JSON file containing the
            configuration to be loaded.

    Returns:
        A dictionary containing the augmented configuration.
    """

    # Make sure that the file_path is an instance of Path
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    # Double-check that the target file exists
    if not file_path.exists():
        raise FileNotFoundError(f'{file_path} does not exist!')

    # Load the config file into a dict
    with open(file_path, 'r') as json_file:
        config: Dict[str, Any] = json.load(json_file)

    # Add the path to the experiments folder to the config dict
    config['experiment_dir'] = file_path.parent or '.'

    return config


def get_datasets_dir() -> Path:
    """
    Get the Path of the datasets directory.

    Returns:
        Path to the datasets directory.
    """

    return get_hsr4hci_dir() / 'datasets'


def get_hsr4hci_dir() -> Path:
    """
    Get path to directory of the hsr4hci package.

    Returns:
        Path to the hsr4hci package directory.
    """

    return Path(hsr4hci.__file__).parent.parent
