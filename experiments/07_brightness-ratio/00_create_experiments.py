"""
Create experiments.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from itertools import product
from pathlib import Path
from typing import List

import argparse
import json
import os
import sys
import time

import click

from hsr4hci.config import load_config, get_hsr4hci_dir
from hsr4hci.htcondor import SubmitFile


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def create_submit_file(
    experiment_dir: Path,
    search_mode: str,
    algorithm: str,
) -> Path:
    """
    Create the HTCondor submit file for an experiment.

    Args:
        experiment_dir: Path to the experiment directory for which to
            create the submit file.
        search_mode: Search mode (blind search or hypothesis-based).
            Required to select the script that runs the experiment.
        algorithm: The algorithm to use (PCA or HSR + signal fitting /
            signal masking).

    Returns:
        Path to the *.sub file that runs the experiment.
    """

    # Define path to scripts directory
    scripts_dir = (
        get_hsr4hci_dir() / 'scripts' / 'experiments' / 'brightness-ratio'
    )

    # Define script and requirements for blind-search PCA / HSR
    if search_mode == 'blind_search':
        if algorithm == 'pca':
            memory = 10240
            cpus = 1
            job_script = scripts_dir / 'run_pca.py'
        elif algorithm in ('signal_fitting', 'signal_masking'):
            memory = 8192
            cpus = 2
            job_script = scripts_dir / 'run_hsr.py'
        else:
            raise ValueError(f'Illegal value "{algorithm}" for algorithm!')

    # Define script and requirements for hypothesis-based HSR
    elif search_mode == 'hypothesis_based':
        if algorithm in ('signal_fitting', 'signal_masking'):
            memory = 8192
            cpus = 2
            job_script = scripts_dir / 'run_hypothesis_hsr.py'
        else:
            raise ValueError(f'Illegal value "{algorithm}" for algorithm!')

    # Raise error for illegal search_mode
    else:
        raise ValueError(f'Illegal value "{search_mode}" for search_mode!')

    # Create the clusterlogs directory
    htcondor_dir = experiment_dir / 'htcondor'
    clusterlogs_dir = htcondor_dir / 'clusterlogs'

    # Create a new submit file and define requirements
    submit_file = SubmitFile(
        clusterlogs_dir=clusterlogs_dir,
        memory=memory,
        cpus=cpus,
        requirements=['Target.CpuFamily =!= 21'],
    )

    # Add the experiment as a job
    submit_file.add_job(
        name='run_experiment',
        job_script=job_script.as_posix(),
        arguments={'experiment-dir': experiment_dir.as_posix()},
        bid=5,
    )

    # Save the submit file in the HTCondor directory
    file_path = htcondor_dir / 'run_experiment.sub'
    submit_file.save(file_path)

    return file_path


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nCREATE EXPERIMENTS FOR THROUGHPUT TABLE\n', flush=True)

    # -------------------------------------------------------------------------
    # Set up parser and get command line arguments
    # -------------------------------------------------------------------------

    # Set up a parser and get the algorithm for which to create experiments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['signal_fitting', 'signal_masking', 'pca'],
        required=True,
        help='Name of the HCIpp algorithm for which to create experiments.',
    )
    parser.add_argument(
        '--search-mode',
        type=str,
        choices=['blind_search', 'hypothesis_based'],
        required=True,
        help='Blind search or hypothesis-based (only relevant for HSR).',
    )
    parser.add_argument(
        '--contrasts',
        nargs='+',
        default=[
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            8.5,
            9.0,
            9.5,
            10.0,
            10.5,
            11.0,
            11.5,
            12.0,
        ],
        type=float,
        help='Contrast values (in mag) for which to create experiments.',
    )
    parser.add_argument(
        '--azimuthal-positions',
        nargs='+',
        default=['a', 'b', 'c', 'd', 'e', 'f'],
        type=str,
        help='Azimuthal positions for which to create experiments.',
    )
    parser.add_argument(
        '--separations',
        nargs='+',
        default=[1, 2, 3, 4, 5, 6, 7, 8],
        type=float,
        help='Separations (in PSF FWHMs) for which to create experiments.',
    )
    parser.add_argument(
        '--no-baseline-experiment',
        action='store_true',
        default=False,
        help='If flag is set, no experiment without fake planets is created.',
    )
    args = parser.parse_args()

    # Define shortcuts to arguments
    algorithm = args.algorithm
    search_mode = args.search_mode
    contrasts = sorted(args.contrasts)
    separations = sorted(args.separations)
    azimuthal_positions = sorted(args.azimuthal_positions)
    create_baseline_experiment = not args.no_baseline_experiment
    n_experiments = int(
        len(contrasts) * len(separations) * len(azimuthal_positions)
    )

    # Define path to this algorithm's base directory and check if it exists
    algorithm_dir = Path('.', search_mode, algorithm).resolve()
    if not algorithm_dir.exists():
        raise RuntimeError(f'{algorithm_dir} does not exist!')

    # Confirm that the parameter space is okay
    print('I will create experiments for the following parameters:')
    print(f'  search_mode:         {search_mode}')
    print(f'  algorithm:           {algorithm}')
    print(f'  contrasts:           {contrasts}')
    print(f'  separations:         {separations}')
    print(f'  azimuthal_positions: {azimuthal_positions}')
    print(
        f'This corresponds to {n_experiments} + '
        f'{int(create_baseline_experiment)} experiments.'
    )
    if not click.confirm('\nDo you want to continue?', default=True):
        sys.exit('')
    print()

    # -------------------------------------------------------------------------
    # Load base configuration and create experiment directory
    # -------------------------------------------------------------------------

    # Load (base) configuration
    file_path = algorithm_dir / 'base_config.json'
    config = load_config(file_path)

    # Create experiments directory for this algorithm
    experiments_dir = algorithm_dir / 'experiments'
    experiments_dir.mkdir(exist_ok=True)

    # Keep track of all the submit files for the experiments
    submit_file_paths: List[Path] = []

    # -------------------------------------------------------------------------
    # Create baseline experiment (no fake planets)
    # -------------------------------------------------------------------------

    # In the baseline experiment, we run our post-processing algorithm on the
    # data stack without any fake planets to estimate the standard deviation
    # of the noise apertures. This is the basis of the contrast curve, which
    # we then need to "calibrate" by correcting for the throughput (which we
    # determine using fake planets). Since we do not inject any fake planets,
    # we only need a single experiment for all separations / contrasts.

    if create_baseline_experiment:

        print('Creating no_fake_planets directory...', end=' ', flush=True)

        # Create the experiment directory
        no_fake_planets_dir = experiments_dir / 'no_fake_planets'
        no_fake_planets_dir.mkdir(exist_ok=True)

        # Store the base configuration (which matches the baseline experiment)
        file_path = no_fake_planets_dir / 'config.json'
        with open(file_path, 'w') as json_file:
            json.dump(config, json_file, indent=2)

        # Create a submit file for the experiment
        file_path = create_submit_file(
            experiment_dir=no_fake_planets_dir,
            search_mode=search_mode,
            algorithm=algorithm,
        )
        submit_file_paths.append(file_path)

        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create experiments that use fake planets
    # -------------------------------------------------------------------------

    # Compute the Cartesian product of the contrasts, the separations and the
    # azimuthal positions -- this is the grid for which we create experiments.
    combinations = list(product(contrasts, separations, azimuthal_positions))

    # Loop over all combinations and create experiments
    for (contrast, separation, azimuthal_position) in combinations:

        # Define experiment name and create the directory for this experiment
        name = f'{contrast:.2f}__{separation:.1f}__{azimuthal_position}'
        experiment_dir = (experiments_dir / name).resolve()
        experiment_dir.mkdir(exist_ok=True, parents=True)

        print(f'Creating experiment_ {experiment_dir}...', end=' ', flush=True)

        # Create experiment configuration (by overwriting previous config)
        config['injection']['separation'] = separation
        config['injection']['contrast'] = contrast
        config['injection']['azimuthal_position'] = azimuthal_position

        # Store the experiment configuration
        file_path = experiment_dir / 'config.json'
        with open(file_path, 'w') as json_file:
            json.dump(config, json_file, indent=2)

        # Create a submit file for the experiment
        file_path = create_submit_file(
            experiment_dir=experiment_dir,
            search_mode=search_mode,
            algorithm=algorithm,
        )
        submit_file_paths.append(file_path)

        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create a shell script that allows to launch the jobs on the cluster
    # -------------------------------------------------------------------------

    print('\nCreating shell script to start jobs...', end=' ', flush=True)
    file_path = algorithm_dir / 'start_jobs.sh'
    with open(file_path, 'w') as sh_file:
        for submit_file_path in submit_file_paths:
            sh_file.write(f'condor_submit_bid 5 {submit_file_path} ;\n')
    os.chmod(file_path, 0o755)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
