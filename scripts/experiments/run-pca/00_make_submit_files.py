"""
Create submit files to run experiment on an HTCondor-based cluster.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import time
import os

from hsr4hci.config import load_config, get_hsr4hci_dir
from hsr4hci.data import load_parang
from hsr4hci.htcondor import SubmitFile, DAGFile


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nMAKE SUBMIT FILES\n', flush=True)

    # -------------------------------------------------------------------------
    # Set up parser to get command line arguments
    # -------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-dir',
        type=str,
        required=True,
        metavar='PATH',
        help='Path to experiment directory.',
    )
    parser.add_argument(
        '--bid',
        type=int,
        default=5,
        help='Amount of cluster dollars to bid for each job.',
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load config and data, get command line variables and define shortcuts
    # -------------------------------------------------------------------------

    # Get experiment directory
    experiment_dir = Path(os.path.expanduser(args.experiment_dir))
    if not experiment_dir.exists():
        raise NotADirectoryError(f'{experiment_dir} does not exist!')

    # Define a couple of path variables
    htcondor_dir = experiment_dir / 'htcondor'
    clusterlogs_dir = htcondor_dir / 'clusterlogs'

    # Define shortcuts to remaining command line arguments
    bid = args.bid

    # Define scripts directory
    scripts_dir = get_hsr4hci_dir() / 'scripts' / 'experiments'

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    config = load_config(experiment_dir / 'config.json')
    print('Done!', flush=True)

    # Load frames, parallactic angles, etc. from HDF file
    print('Loading data set...', end=' ', flush=True)
    parang = load_parang(**config['dataset'])
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Compute the expected memory consumption of a training job
    # -------------------------------------------------------------------------

    # Metadata of the data set
    n_frames = len(parang)
    frame_size = config['dataset']['frame_size']

    # Compute the memory required for loading the full stack (in bytes)
    stack_memory = n_frames * frame_size[0] * frame_size[1] * 8

    # Estimate memory needed to perform PCA on stack of this size
    # FIXME: Improve this very crude estimate...
    expected_job_memory = 32 * int(stack_memory / 1024 ** 2) + 2048

    # -------------------------------------------------------------------------
    # Instantiate a new DAG file
    # -------------------------------------------------------------------------

    print('Instantiating new DAG file...', end=' ', flush=True)
    dag_file = DAGFile()
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create a submit file for running the PCA
    # -------------------------------------------------------------------------

    name = '01_run_pca'

    # Create submit file and add job
    print(f'Creating {name}.sub...', end=' ', flush=True)
    submit_file = SubmitFile(
        clusterlogs_dir=clusterlogs_dir.as_posix(),
        memory=expected_job_memory,
        cpus=4,
    )
    submit_file.add_job(
        name=name,
        job_script=(scripts_dir / 'run-pca' / f'{name}.py').as_posix(),
        arguments={
            'experiment-dir': experiment_dir.as_posix(),
        },
        bid=bid,
        queue=1,
    )
    file_path = htcondor_dir / f'{name}.sub'
    submit_file.save(file_path=file_path.as_posix())
    print('Done!', flush=True)

    # Add submit file to DAG
    print(f'Adding {name}.sub to DAG file...', end=' ', flush=True)
    dag_file.add_submit_file(
        name=name,
        attributes=dict(file_path=file_path.as_posix(), bid=bid),
    )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create a submit file for creating a plot of the signal estimate
    # -------------------------------------------------------------------------

    name = 'evaluate_and_plot_signal_estimate'

    print(f'Creating {name}.sub...', end=' ', flush=True)
    submit_file = SubmitFile(
        clusterlogs_dir=clusterlogs_dir.as_posix(),
        memory=1024,
        cpus=1,
    )
    submit_file.add_job(
        name=name,
        job_script=(
            scripts_dir / 'evaluate-and-plot' / f'{name}.py'
        ).as_posix(),
        arguments={
            'experiment-dir': experiment_dir.as_posix(),
        },
        bid=bid,
    )
    file_path = htcondor_dir / f'{name}.sub'
    submit_file.save(file_path=file_path)
    print('Done!', flush=True)

    # Add submit file to DAG
    print(f'Adding {name}.sub to DAG file...', end=' ', flush=True)
    dag_file.add_submit_file(
        name=name,
        attributes=dict(file_path=file_path.as_posix(), bid=bid),
    )
    dag_file.add_dependency('01_run_pca', name)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Save the DAG file
    # -------------------------------------------------------------------------

    print('Saving DAG file...', end=' ', flush=True)
    file_path = htcondor_dir / 'run_experiment.dag'
    dag_file.save(file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
