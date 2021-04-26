"""
Create experiments.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import json
import os
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
        '--n-splits',
        type=int,
        default=64,
        help=(
            'Number of splits into which the training data is divided to '
            'parallelize the training.'
        ),
    )
    parser.add_argument(
        '--bid',
        type=int,
        default=5,
        help='Amount of cluster dollars to bid for each job.',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
    )
    args = parser.parse_args()

    # Get arguments
    algorithm = args.algorithm
    dataset = args.dataset
    n_splits = args.n_splits
    bid = args.bid

    # -------------------------------------------------------------------------
    # Create experiments
    # -------------------------------------------------------------------------

    # Define binning factors for which we want to create an experiment
    # Note: We do not need 1 here, because we will symlink to the experiment
    #       from 01_first-results to avoid computing anything twice. The
    #       symlinking has to happen manually (for now).
    factors = (
        2, 3, 4, 5, 6, 8, 10, 16, 25, 32, 64, 128, 150, 200, 300, 400, 500
    )

    # Read in the basic experiment configuration
    # We basically copy over the version from "factor_1", which is symlinked
    # to the 01_first-results directory (see above), and then only change the
    # binning factor in the "dataset" section of the configuration.
    main_dir = (
        get_hsr4hci_dir()
        / 'experiments'
        / '02_temporal-binning'
        / dataset
        / algorithm
    )
    file_path = main_dir / 'factor_1' / 'config.json'
    experiment_config = load_config(file_path)

    # Define the directory with the script for creating the submit files
    if algorithm == 'pca':
        scripts_dir = get_hsr4hci_dir() / 'scripts' / 'experiments' / 'run-pca'
        n_splits_argument = ''
    else:
        scripts_dir = (
            get_hsr4hci_dir() / 'scripts' / 'experiments' / 'multiple-scripts'
        )
        n_splits_argument = f'--n-splits {n_splits}'

    # Keep track of lines that will be written to shell scripts
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
            f'--bid {bid} '
            f'{n_splits_argument} '
            f'--experiment-dir {experiment_dir.as_posix()} ;'
        )

        # Add line for the shell script to submit the DAGs
        file_path = experiment_dir / 'htcondor' / 'run_experiment.dag'
        start_jobs.append(f'condor_submit_dag {file_path.as_posix()}')

    # Create the shell script to create submit files for all experiments;
    # use chmod() to make the script executable
    file_path = main_dir / '00_create_submit_files.sh'
    with open(file_path, 'w') as sh_file:
        sh_file.write('#!/bin/zsh\n\n')
        for line in create_submit_files:
            sh_file.write(f'{line}\n')
    os.chmod(file_path, 0o755)

    # Create the shell script to launch the experiments on the cluster;
    # use chmod() to make the script executable
    file_path = main_dir / '01_start_jobs.sh'
    with open(file_path, 'w') as sh_file:
        sh_file.write('#!/bin/zsh\n\n')
        for line in start_jobs:
            sh_file.write(f'{line}\n')
    os.chmod(file_path, 0o755)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
