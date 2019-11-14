"""
Generate experiments to study the influence of certain parameters.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from copy import deepcopy
from pathlib import Path

import json
import os
import shutil
import time


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nGENERATE EXPERIMENTS\n', flush=True)

    # -------------------------------------------------------------------------
    # Load base configuration
    # -------------------------------------------------------------------------

    with open('./base_config.json', 'r') as json_file:
        base_config = json.load(json_file)

    # -------------------------------------------------------------------------
    # Create a folder in the experiments directory for experiments we generate
    # -------------------------------------------------------------------------

    experiments_dir = os.path.abspath('../02_experiments')
    Path(experiments_dir).mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Loop over parameter combinations and create experiments
    # -------------------------------------------------------------------------

    list_of_experiments = list()

    for weight_mode in ['default', 'weighted', 'train_test']:

        # ---------------------------------------------------------------------
        # Create experiment config as a dictionary
        # ---------------------------------------------------------------------

        # Create a copy of the base_config
        config = deepcopy(base_config)

        # Set weight_mode parameter
        config['experiment']['model']['weight_mode'] = weight_mode

        # ---------------------------------------------------------------------
        # Set up an experiment folder with this configuration
        # ---------------------------------------------------------------------

        # Create the name for this experiment
        experiment_name = f'weight-mode_{weight_mode}'

        # Create a folder with that name
        experiment_dir = os.path.join(experiments_dir, experiment_name)
        Path(experiment_dir).mkdir(exist_ok=True)
        list_of_experiments.append(experiment_name)

        # Copy Python scripts from default_cluster experiments
        default_cluster_dir = '../../../default/cluster'
        python_scripts = [_ for _ in os.listdir(default_cluster_dir)
                          if _.endswith('.py')]
        for python_script in python_scripts:
            shutil.copy(os.path.join(default_cluster_dir, python_script),
                        experiment_dir)

        # Save the generated experiment configuration to a JSON file
        file_path = os.path.join(experiment_dir, 'config.json')
        with open(file_path, 'w') as json_file:
            json.dump(config, json_file, indent=2)

        print(f'Created experiment: {experiment_name}')

    # -------------------------------------------------------------------------
    # Save list of experiments
    # -------------------------------------------------------------------------

    file_path = os.path.join(experiments_dir, 'list_of_experiments.txt')
    with open(file_path, 'w') as text_file:
        text_file.write('\n'.join(sorted(list_of_experiments)))
        text_file.write('\n')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
