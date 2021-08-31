"""
Create experiments to study effect of temporal binning.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import json
import os
import time

import numpy as np

from hsr4hci.config import load_config, get_hsr4hci_dir


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCREATE EXPERIMENTS FOR TEMPORAL BINNING\n', flush=True)

    # -------------------------------------------------------------------------
    # Set up parser and get command line arguments
    # -------------------------------------------------------------------------

    # Set up argument parser and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['pca', 'signal_fitting', 'signal_masking'],
        required=True,
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
    )
    args = parser.parse_args()

    # Define shortcuts for arguments
    algorithm = args.algorithm
    dataset = args.dataset

    # -------------------------------------------------------------------------
    # Define shortcuts and global constants
    # -------------------------------------------------------------------------

    # Define binning factors (we exclude "1" here, because we can re-use the
    # results for 01_first-results)
    binning_factors = np.unique(np.geomspace(2, 2000, 21).astype(int))

    # Define the directory that contains the script for creating the submit
    # files for the experiments; fix number of jobs for the HSR experiments
    scripts_dir = get_hsr4hci_dir() / 'scripts' / 'experiments'
    if algorithm == 'pca':
        scripts_dir /= 'run-pca'
        n_splits_argument = ''
    else:
        scripts_dir /= 'multiple-scripts'
        n_splits_argument = '--n-splits 128 '

    # -------------------------------------------------------------------------
    # Create experiments folder, symlink factor_1, and read in base config
    # -------------------------------------------------------------------------

    # Create the directory for the given combination of dataset and algorithm
    main_dir = (
        get_hsr4hci_dir()
        / 'experiments'
        / '02_temporal-binning'
        / dataset
        / algorithm
    )
    print(f'Creating directory: {main_dir} ...', end='', flush=True)
    main_dir.mkdir(exist_ok=True, parents=True)
    print('Done!', flush=True)

    # Create the experiment with binning factor = 1. This should just be a
    # symlink to the respective directory 01_first-results so that we do not
    # have to run the most expensive experiment twice.
    print('Creating experiment: binning_factor-1 ...', end='', flush=True)
    src_dir = (
        get_hsr4hci_dir()
        / 'experiments'
        / '01_first-results'
        / dataset
        / algorithm
    )
    dst_dir = main_dir / 'binning_factor-1'
    os.symlink(src=src_dir, dst=dst_dir)
    print('Done! (symlinked)', flush=True)

    # Read in the base experiment configuration: We basically copy the config
    # from the (symlinked) "factor_1" directory, and then only change the
    # binning factor in the "dataset" section of the configuration.
    file_path = main_dir / 'factor_1' / 'config.json'
    experiment_config = load_config(file_path)

    # -------------------------------------------------------------------------
    # Create experiment for each binning factor
    # -------------------------------------------------------------------------

    # Keep track of lines that will be written to shell scripts
    create_submit_files = []
    start_jobs = []

    # Loop over different binning factors and create experiment folders
    for binning_factor in binning_factors:

        print(
            f'Creating experiment: binning_factor-{binning_factor} ...',
            end='',
            flush=True,
        )

        # Update the experiment configuration to the current binning factor
        experiment_config['dataset']['binning_factor'] = binning_factor

        # Create the experiment folder
        experiment_dir = main_dir / f'binning_factor-{binning_factor}'
        experiment_dir.mkdir(exist_ok=True)

        # Save the experiment configuration
        file_path = experiment_dir / 'config.json'
        with open(file_path, 'w') as json_file:
            json.dump(experiment_config, json_file)

        # Add line for the shell script to create the submit files
        file_path = scripts_dir / '00_make_submit_files.py'
        create_submit_files.append(
            f'python {file_path.as_posix()} '
            f'--bid 5 '
            f'{n_splits_argument}'
            f'--experiment-dir {experiment_dir.as_posix()} ;'
        )

        # Add line for the shell script to submit the DAGs
        file_path = experiment_dir / 'htcondor' / 'run_experiment.dag'
        start_jobs.append(f'condor_submit_dag {file_path.as_posix()}')

        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create shel scripts for creating submit file and starting the jobs
    # -------------------------------------------------------------------------

    # Create the shell script to create submit files for all experiments;
    # use chmod() to make the script executable
    print('Creating 00_create_submit_files.sh ...', end='', flush=True)
    file_path = main_dir / '00_create_submit_files.sh'
    with open(file_path, 'w') as sh_file:
        sh_file.write('#!/bin/zsh\n\n')
        for line in create_submit_files:
            sh_file.write(f'{line}\n')
    os.chmod(file_path, 0o755)
    print('Done!', flush=True)

    # Create the shell script to launch the experiments on the cluster;
    # use chmod() to make the script executable
    print('Creating 01_start_jobs.sh ...', end='', flush=True)
    file_path = main_dir / '01_start_jobs.sh'
    with open(file_path, 'w') as sh_file:
        sh_file.write('#!/bin/zsh\n\n')
        for line in start_jobs:
            sh_file.write(f'{line}\n')
    os.chmod(file_path, 0o755)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
