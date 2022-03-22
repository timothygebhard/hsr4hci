"""
Create experiments based on a given base_config.json and the command
line arguments passed to this script.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from itertools import product
from pathlib import Path
from typing import List

import argparse
import json
import sys
import time

from tqdm.auto import tqdm

import click

from hsr4hci.config import load_config, get_hsr4hci_dir
from hsr4hci.htcondor import SubmitFile


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def create_submit_file(
    experiment_dir: Path,
    algorithm: str,
    memory: int = 8192,
) -> Path:
    """
    Create the HTCondor submit file for an experiment.

    Args:
        experiment_dir: Path to the experiment directory for which to
            create the submit file.
        algorithm: The algorithm to use: "pca" or "signal_fitting" or
            "signal_masking".
        memory: Memory (in MB) to require for experiments.

    Returns:
        Path to the *.sub file that runs the experiment.
    """

    # Define path to scripts directory
    scripts_dir = (
        get_hsr4hci_dir()
        / 'scripts'
        / 'experiments'
        / 'brightness-ratio'
    )

    # Define script and requirements for different algorithms
    if algorithm == 'pca':
        memory += 2048
        cpus = 1
        job_script = scripts_dir / 'run_pca.py'
    elif algorithm == 'hsr':
        cpus = 2
        job_script = scripts_dir / 'run_hsr.py'
    else:
        raise ValueError(f'Illegal value "{algorithm}" for algorithm!')

    # Create the clusterlogs directory
    htcondor_dir = experiment_dir / 'htcondor'
    clusterlogs_dir = htcondor_dir / 'clusterlogs'

    # Create a new submit file and define requirements
    submit_file = SubmitFile(
        clusterlogs_dir=clusterlogs_dir,
        memory=memory,
        cpus=cpus,
        requirements=[
            'Target.CpuFamily =!= 21',
            'Target.Machine =!= "g095.internal.cluster.is.localnet"',
        ],
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
        '--directory',
        type=str,
        required=True,
        help='Main directory of the experiment set.',
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
        default=[2, 3, 4, 5, 6, 7, 8],
        type=float,
        help='Separations (in PSF FWHMs) for which to create experiments.',
    )
    parser.add_argument(
        '--no-baseline-experiment',
        action='store_true',
        default=False,
        help='If flag is set, no experiment without fake planets is created.',
    )
    parser.add_argument(
        '--memory',
        type=int,
        default=8192,
        help='Memory to require for experiments (use to overwrite defaults).',
    )
    args = parser.parse_args()

    # Define shortcuts to arguments
    main_dir = Path(args.directory).resolve()
    contrasts = sorted(args.contrasts)
    separations = sorted(args.separations)
    azimuthal_positions = sorted(args.azimuthal_positions)
    create_baseline_experiment = not args.no_baseline_experiment
    memory = int(args.memory)
    n_experiments = int(
        len(contrasts) * len(separations) * len(azimuthal_positions)
    )

    # Make sure the main directory (where the base_config.json resides) exists
    if not main_dir.exists():
        raise RuntimeError(f'{main_dir} does not exist!')

    # -------------------------------------------------------------------------
    # Load base configuration and confirm creation of experiments
    # -------------------------------------------------------------------------

    # Load (base) configuration
    file_path = main_dir / 'base_config.json'
    config = load_config(file_path)

    # Define some shortcuts
    algorithm = config['algorithm']
    algorithm_extra = f'({config["train_mode"]})' if algorithm == 'hsr' else ''

    # Confirm that the parameter space is okay
    print('I will create experiments for the following parameters:')
    print(f'  algorithm:           {algorithm.upper()} {algorithm_extra}')
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

    # Create experiments directory for this algorithm
    experiments_dir = main_dir / 'experiments'
    experiments_dir.mkdir(exist_ok=True)

    # Keep track of all the submit files for the experiments
    submit_file_paths: List[Path] = []

    # We will create a DAG file with the submit files for all experiments so
    # that we can easily launch all jobs at once. We now define the path to
    # the file and make sure it is empty.
    dag_file_path = main_dir / 'start_jobs.dag'
    open(dag_file_path, 'w').close()

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
            algorithm=algorithm,
            memory=memory,
        )
        submit_file_paths.append(file_path)

        # Add the baseline experiment as a job to the DAG file
        with open(dag_file_path, 'a') as dag_file:
            dag_file.write(f'JOB no_fake_planets {file_path}\n')

        print('Done!\n', flush=True)

    # -------------------------------------------------------------------------
    # Create experiments that use fake planets
    # -------------------------------------------------------------------------

    print('Creating experiment with fake planets:', flush=True)

    # Compute the Cartesian product of the contrasts, the separations and the
    # azimuthal positions -- this is the grid for which we create experiments.
    combinations = tqdm(
        iterable=list(product(contrasts, separations, azimuthal_positions)),
        ncols=80
    )

    # Open the DAG file where we keep track of all experiments
    with open(dag_file_path, 'a') as dag_file:

        # Loop over all combinations (with progress bar) and create experiments
        for (contrast, separation, azimuthal_position) in combinations:

            # Define experiment name and create experiment directory
            name = f'{contrast:.2f}__{separation:.1f}__{azimuthal_position}'
            experiment_dir = (experiments_dir / name).resolve()
            experiment_dir.mkdir(exist_ok=True, parents=True)

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
                algorithm=algorithm,
                memory=memory,
            )
            submit_file_paths.append(file_path)

            # Add the submit file to the DAG file
            dag_file.write(f'JOB {name} {file_path}\n')

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
