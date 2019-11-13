"""
Generate experiments to study the influence of certain parameters.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from copy import deepcopy
from itertools import product
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

    with open('base_config.json', 'r') as json_file:
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

    for fit_intercept, subtract_median, orthogonalize, pca, add_planet_col in \
            product((True, False), repeat=5):

        # ---------------------------------------------------------------------
        # Create experiment config as a dictionary
        # ---------------------------------------------------------------------

        # Create a copy of the base_config
        config = deepcopy(base_config)

        # Set fit_intercept parameter
        config['experiment']['model']['parameters']['fit_intercept'] = \
            fit_intercept

        # Set pre-subtraction parameters
        config['dataset']['presubtract'] = \
            'median' if subtract_median else None

        # Add orthogonalization to pre-processing of sources
        if orthogonalize:
            config['experiment']['sources']['preprocessing'].append(
                {
                    "type": "orthogonalize",
                    "parameters": {
                        "target": "planet_signal"
                    }
                }
            )

        # Add PCA to pre-processing of sources
        if pca:
            config['experiment']['sources']['preprocessing'].append(
                {
                    "type": "pca",
                    "parameters": {
                        "pca_mode": "temporal",
                        "n_components": 32,
                        "sv_power": 1.0
                    }
                }
            )

        # Set add_planet_column parameter
        config['experiment']['add_planet_column'] = add_planet_col

        # ---------------------------------------------------------------------
        # Set up an experiment folder with this configuration
        # ---------------------------------------------------------------------

        # Create the name for this experiment
        experiment_name = (f'fit_intercept-{int(fit_intercept)}__'
                           f'presubtract_median-{int(subtract_median)}__'
                           f'compute_pca-{int(pca)}__'
                           f'orthogonalize-{int(orthogonalize)}__'
                           f'add_planet_column-{int(add_planet_col)}')

        # Create a folder with that name
        experiment_dir = os.path.join(experiments_dir, experiment_name)
        Path(experiment_dir).mkdir(exist_ok=True)
        list_of_experiments.append(experiment_dir)

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
