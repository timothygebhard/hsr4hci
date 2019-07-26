"""
Utilities for reading in config files.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import json
import os
import socket


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def load_config(config_file_path: str) -> dict:

    # Build the full path to the config file and check if it exists
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f'{config_file_path} does not exist!')

    # Load the config file into a dict
    with open(config_file_path, 'r') as json_file:
        config = json.load(json_file)

    # -------------------------------------------------------------------------
    # Amend configuration
    # -------------------------------------------------------------------------

    # Add the path to the experiments folder to the config dict
    config['experiment_dir'] = os.path.basename(config_file_path)

    # Add implicitly defined variables
    config['dataset']['x_center'] = config['dataset']['x_size'] / 2
    config['dataset']['y_center'] = config['dataset']['y_size'] / 2
 
    # Replace dummy data directory with machine-specific data directory
    data_dir = get_data_dir()
    config['dataset']['file_path'] = \
        config['dataset']['file_path'].replace('DATA_DIR', data_dir)
 
    return config


def get_data_dir():

    hostname = str(socket.gethostname())

    if hostname == 'login2':
        return '/is/cluster/tgebhard/datasets/exoplanets/Markus'

    elif hostname == 'Markuss-Macbook-Pro.local':
        return ''

    else:
        raise ValueError(f'Hostname "{hostname}" not known!')
