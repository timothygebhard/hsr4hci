"""
Utilities for reading in config files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import getpass
import json
import os
import re
import socket


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def load_config(config_file_path: str) -> dict:
    """
    Load and amend an experiment configuration.

    Args:
        config_file_path: Path to the JSON file containing the
            configuration to be loaded.

    Returns:
        A dictionary containing the amended configuration.
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
    # Amend configuration (i.e., add implicitly defined variables)
    # -------------------------------------------------------------------------

    # Add the path to the experiments folder to the config dict
    config['experiment_dir'] = os.path.dirname(config_file_path)

    # Add implicitly defined variables
    config['dataset']['frame_center'] = \
        tuple(map(lambda x: x / 2, config['dataset']['frame_size']))
 
    # Replace dummy data directory with machine-specific data directory
    data_dir = get_data_dir()
    config['dataset']['file_path'] = \
        config['dataset']['file_path'].replace('DATA_DIR', data_dir)
 
    return config


def get_data_dir() -> str:
    """
    Get the path to the data directory (based on the current host name).

    Returns:
        Path to the data directory.
    """

    # TODO: Getting the data directory based on the current hostname (and
    #       possibly the username) seems increasingly messy (see below).
    #       Can we maybe find a better solution here?

    # Get current hostname and username
    hostname = str(socket.gethostname())
    username = str(getpass.getuser())

    # -------------------------------------------------------------------------
    # Cluster at the MPI-IS
    # -------------------------------------------------------------------------

    # On the MPI cluster, there are two types of nodes:
    #   1. login nodes (names "login1" and "login2)
    #   2. worker nodes (naming scheme: [e|g|o|t] + 3 digits; e.g. "g002")
    if bool(re.match(r'^login1$|^login2$|^[egot]\d{3}$', hostname)):
        return '/is/cluster/tgebhard/datasets/exoplanets/'

    # -------------------------------------------------------------------------
    # Markus's computers
    # -------------------------------------------------------------------------

    elif hostname == 'bluesky':
        return '/net/ipa-gate.phys.ethz.ch/export/ipa/meyer/hsr'

    elif (hostname == 'Markuss-MacBook-Pro.local' or
          hostname == 'Markuss-MBP'):
        return ''

    # -------------------------------------------------------------------------
    # Timothy's computers
    # -------------------------------------------------------------------------

    # If connected to "normal" networks
    elif hostname == 'Timothys-MacBook-Pro.local':
        return os.path.expanduser('~/Documents/PhD/datasets/exoplanets')

    # If connected to the ETH wifi (which changes the hostname)
    elif bool(re.match(r'^\S+.ethz.ch$', hostname)) and username == 'timothy':
        return os.path.expanduser('~/Documents/PhD/datasets/exoplanets')

    # -------------------------------------------------------------------------
    # Default: Raise ValueError
    # -------------------------------------------------------------------------

    else:
        raise ValueError(f'Hostname "{hostname}" not known!')
