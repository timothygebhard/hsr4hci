"""
Methods for reading in configuration files and getting paths to main
`hsr4hci` subdirectories.
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
    Load and augment a (JSON) configuration file.

    Args:
        file_path: Path to the JSON file containing the
            configuration to be loaded.

    Returns:
        A dictionary containing configuration.
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

    return config


def get_hsr4hci_dir() -> Path:
    """
    Get path to directory of the hsr4hci package.

    Returns:
        Path to the hsr4hci package directory.
    """

    return Path(hsr4hci.__file__).parent.parent


def get_datasets_dir() -> Path:
    """
    Get the Path of the `datasets` directory.

    Returns:
        Path to the `datasets` directory.
    """

    return get_hsr4hci_dir() / 'datasets'


def get_experiments_dir() -> Path:
    """
    Get the Path of the `experiments` directory.

    Returns:
        Path to the `experiments` directory.
    """

    return get_hsr4hci_dir() / 'experiments'
