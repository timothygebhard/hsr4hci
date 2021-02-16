"""
Utilities for reading in config files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, Union

import json
import os

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


def load_dataset_config(target_name: str, filter_name: str) -> Dict[str, Any]:
    """

    Args:
        target_name:
        filter_name:

    Returns:

    """

    # Construct full path to the JSON file containing the configuration
    # of the target data set
    file_name = f'{target_name}__{filter_name}.json'.lower()
    file_path = Path(hsr4hci.__file__).parent.parent / 'datasets' / file_name

    # Double-check that the target file exists
    if not file_path.exists():
        raise FileNotFoundError(
            f'No data set configuration file found for target "{target_name}" '
            f'and filter "{filter_name}"!'
        )

    # Otherwise, read in the JSON file, parse it to a dict, and return it
    with open(file_path, 'r') as json_file:
        dataset_config: Dict[str, Any] = json.load(json_file)
    return dataset_config


def get_data_dir() -> Path:
    """
    Get the Path of the data directory from an environment variable.

    Returns:
        Path to the data directory.
    """

    # Check if the HSR4HCI_DATA_DIR environment variable is set
    if 'HSR4HCI_DATA_DIR' in os.environ.keys():
        data_dir = Path(os.environ['HSR4HCI_DATA_DIR'])
    else:
        raise RuntimeError('Environment variable HSR4HCI_DATA_DIR not set!')

    # Check if the value it contains is a valid directory
    if not data_dir.is_dir():
        raise RuntimeError('Value of HSR4HCI_DATA_DIR is not a directory!')

    return data_dir


def get_hsr4hci_dir() -> Path:
    """
    Get path to directory of the hsr4hci package.

    Returns:
        Path to the hsr4hci package directory.
    """

    return Path(hsr4hci.__file__).parent.parent
