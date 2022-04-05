"""
Methods for reading in configuration files and for getting paths to the
directories that contain the data and the experiments.
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
    Load a (JSON) configuration file.

    Args:
        file_path: Path to the JSON file containing the configuration
            to be loaded.

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
    Get path to directory of the ``hsr4hci`` package directory.

    Returns:
        Path to the ``hsr4hci`` package directory.
    """

    return Path(hsr4hci.__file__).parent.parent


def get_datasets_dir() -> Path:
    """
    Get the path of the ``datasets`` directory (i.e., the directory
    where the methods from :class:`hsr4hci.data` will look for data
    sets by default).

    .. note::

        This path needs to be defined in an environmental variable
        called ``HSR4HCI_DATASETS_DIR``, which can be set as follows:

        .. code-block:: bash

            export HSR4HCI_DATASETS_DIR="/path/to/datasets/directory"

        Include this line in your ``.bashrc`` (or similar) to set it
        automatically.

    Returns:
        Path to the ``datasets`` directory.
    """

    # If HSR4HCI_DATASETS_DIR is not set, raise an error
    if (datasets_dir_str := os.getenv('HSR4HCI_DATASETS_DIR')) is None:
        raise KeyError(
            'Environmental variable: HSR4HCI_DATASETS_DIR not defined!'
        )

    # Convert HSR4HCI_DATASETS_DIR to a Path and verify that it exists
    datasets_dir = Path(datasets_dir_str).resolve()
    if not datasets_dir.exists():
        raise NotADirectoryError(f'{datasets_dir} does not exist!')

    return datasets_dir


def get_experiments_dir() -> Path:
    """
    Get the path of the ``experiments`` directory.

    .. note::

        This path needs to be defined in an environmental variable
        called ``HSR4HCI_EXPERIMENTS_DIR``, which can be set as follows:

        .. code-block:: bash

            export HSR4HCI_EXPERIMENTS_DIR="/path/to/experiments/directory"

        Include this line in your ``.bashrc`` (or similar) to set it
        automatically.

    Returns:
        Path to the ``experiments`` directory.
    """

    # If HSR4HCI_EXPERIMENTS_DIR is not set, raise an error
    if (experiments_dir_str := os.getenv('HSR4HCI_EXPERIMENTS_DIR')) is None:
        raise KeyError(
            'Environmental variable: HSR4HCI_EXPERIMENTS_DIR not defined!'
        )

    # Convert HSR4HCI_EXPERIMENTS_DIR to a Path and verify that it exists
    experiments_dir = Path(experiments_dir_str).resolve()
    if not experiments_dir.exists():
        raise NotADirectoryError(f'{experiments_dir} does not exist!')

    return experiments_dir
