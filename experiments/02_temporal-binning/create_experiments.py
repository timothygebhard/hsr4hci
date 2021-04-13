"""
Create experiments.
"""


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import json
import time

from hsr4hci.config import load_config, get_hsr4hci_dir

# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCREATE EXPERIMENTS FOR 02_TEMPORAL-BINNING\n', flush=True)

    # -------------------------------------------------------------------------
    # Set up parser and get command line arguments
    # -------------------------------------------------------------------------

    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['pca', 'signal_fitting', 'signal_masking'],
        required=True,
    )
    parser.add_argument(
        '--dataset', type=str, choices=['beta_pictoris__lp'], required=True
    )
    args = parser.parse_args()

    # Get arguments
    algorithm = args.algorithm
    dataset = args.dataset

    # -------------------------------------------------------------------------
    # Create experiments
    # -------------------------------------------------------------------------

    # Define binning factors for which we want to create an experiment
    # Note: We do not need 1 here, because we will symlink to the experiment
    #       from 01_first-results to avoid computing anything twice
    factors = (2, 3, 4, 5, 6, 8, 10, 16, 25, 32, 64, 128)

    # Read in the basic experiment configuration
    main_dir = (
        get_hsr4hci_dir()
        / 'experiments'
        / '02_temporal-binning'
        / algorithm
        / dataset
    )
    file_path = main_dir / 'factor_1' / 'config.json'
    experiment_config = load_config(file_path)

    # Keep track of lines that will be written to shell scripts
    if algorithm == 'pca':
        scripts_dir = get_hsr4hci_dir() / 'scripts' / 'experiments' / 'run-pca'
    else:
        raise NotImplementedError
    create_submit_files = []
    start_jobs = []

    # Loop over different binning factors and create experiment folders
    for factor in factors:

        # Update the experiment configuration to the current binning factor
        experiment_config['dataset']['binning_factor'] = factor

        # Create the experiment folder
        experiment_dir = main_dir / f'factor_{factor}'
        experiment_dir.mkdir(exist_ok=True)

        # Save the experiment configuration
        file_path = experiment_dir / 'config.json'
        with open(file_path, 'w') as json_file:
            json.dump(experiment_config, json_file)

        # Add line for the shell script to create the submit files
        file_path = scripts_dir / '00_make_submit_files.py'
        create_submit_files.append(
            f'python {file_path.as_posix()} '
            f'--experiment-dir {experiment_dir.as_posix()} ;'
        )

        # Add line for the shell script to submit the DAGs
        file_path = experiment_dir / 'htcondor' / 'run_experiment.dag'
        start_jobs.append(f'csd 5 {file_path.as_posix()}')

    # Create the shell scripts
    with open('create_submit_files.sh', 'w') as sh_file:
        sh_file.write('#!/bin/zsh\n\n')
        for line in create_submit_files:
            sh_file.write(f'{line}\n')
    with open('start_jobs.sh', 'w') as sh_file:
        sh_file.write('#!/bin/zsh\n\n')
        for line in start_jobs:
            sh_file.write(f'{line}\n')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
