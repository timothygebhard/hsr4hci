"""
Create submit files to run experiment on an HTCondor-based cluster.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from sys import getsizeof
from pathlib import Path
from typing import Any

import argparse
import time
import os

from astropy.units import Quantity

import numpy as np

from hsr4hci.config import load_config
from hsr4hci.data import load_dataset
from hsr4hci.htcondor import SubmitFile, DAGFile
from hsr4hci.masking import get_roi_mask
from hsr4hci.units import set_units_for_instrument


# -----------------------------------------------------------------------------
# FUNCTIONS DEFINITIONS
# -----------------------------------------------------------------------------

def get_arguments() -> argparse.Namespace:
    """
    Parse command line arguments that are passed to the script.

    Returns:
        The command line arguments as a Namespace object.
    """

    # Set up parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        '--bid',
        type=int,
        default=50,
        help='Amount of cluster dollars to bid for each job.',
    )
    parser.add_argument(
        '--n-splits',
        type=int,
        default=1,
        help=(
            'Number of splits into which the training data is divided to '
            'parallelize the training.'
        ),
    )

    return parser.parse_args()


def get_size(_object: Any) -> int:
    """
    Auxiliary function to determine the size (in memory) of an object.
    """

    if isinstance(_object, np.ndarray):
        return int(_object.nbytes)
    return getsizeof(_object)


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
    # Load config and data, get command line variables and define shortcuts
    # -------------------------------------------------------------------------

    # Define a couple of path variables
    experiment_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    htcondor_dir = experiment_dir / 'htcondor'
    clusterlogs_dir = htcondor_dir / 'clusterlogs'

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    config = load_config(experiment_dir / 'config.json')
    print('Done!', flush=True)

    # Load frames, parallactic angles, etc. from HDF file
    print('Loading data set...', end=' ', flush=True)
    stack, parang, psf_template, observing_conditions, metadata = load_dataset(
        **config['dataset']
    )
    print('Done!', flush=True)

    # Parse command line arguments and add shortcuts
    args = get_arguments()
    bid = args.bid
    n_splits = args.n_splits

    # -------------------------------------------------------------------------
    # Shortcuts, activate unit conversions, get ROI mask
    # -------------------------------------------------------------------------

    # Metadata of the data set
    n_frames = len(stack)
    pixscale = float(metadata['PIXSCALE'])
    lambda_over_d = float(metadata['LAMBDA_OVER_D'])

    set_units_for_instrument(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
        verbose=False,
    )

    # Define a mask for the ROI
    roi_mask = get_roi_mask(
        mask_size=stack.shape[1:],
        inner_radius=Quantity(*config['roi_mask']['inner_radius']),
        outer_radius=Quantity(*config['roi_mask']['outer_radius']),
    )

    # Compute the expected number of pixels per training job
    n_pixels_per_job = int(np.ceil(np.sum(roi_mask) / n_splits))

    # -------------------------------------------------------------------------
    # Compute the expected memory consumption of a training job
    # -------------------------------------------------------------------------

    # Compute the size of the full data set which we will be loading
    dataset_memory = sum(
        [
            get_size(stack),
            get_size(parang),
            get_size(psf_template),
            get_size(observing_conditions),
            get_size(metadata),
        ]
    )

    print('\nMemory consumption:')
    print(f'-- stack:        {int(get_size(stack) / 1024**2):>4d} MB')
    print(f'-- parang:       {int(get_size(parang) / 1024):>4d} KB')
    print(f'-- psf_template: {int(get_size(psf_template) / 1024):>4d} KB')
    print(f'-- obs_con:      {int(get_size(observing_conditions)):>4d} B')
    print(f'-- metadata:     {int(get_size(metadata)):>4d} B')

    # Compute the size of a single "ROI subset" variable
    array_memory = get_size(np.ones((n_frames, n_pixels_per_job)))

    # Count the number of such variables that we need to keep in memory and
    # write out to the (partial) HDF files
    n_arrays = 2 + 3 * (config['n_signal_times'] + 1)

    # Compute the expected amount of memory that we need per job (in MB)
    expected_job_memory = 4 * dataset_memory + n_arrays * array_memory
    expected_job_memory /= 1024 ** 2
    expected_job_memory *= 2.2
    expected_job_memory = int(expected_job_memory)

    # Compute the expected total memory needed for merging the HDF files
    expected_total_memory = n_arrays * array_memory * n_splits
    expected_total_memory /= 1024 ** 2
    expected_total_memory *= 2
    expected_total_memory = int(expected_total_memory)

    print(
        f'\nPixels per job: {np.sum(roi_mask)} / {n_splits} <= '
        f'{n_pixels_per_job}\n'
    )
    print(f'Data set memory: {int(dataset_memory / 1024 ** 2):6d} MB')
    print(
        f'Expected memory: {expected_job_memory:6d} MB per job\n'
        f'                 {expected_total_memory:6d} MB in total\n'
    )

    # Round up (it doesn't make sense to ask for less than 1 GB on the cluster)
    expected_job_memory = max(expected_job_memory, 1024)
    expected_total_memory = max(expected_total_memory, 1024)

    # -------------------------------------------------------------------------
    # Instantiate a new DAG file
    # -------------------------------------------------------------------------

    print('Instantiating new DAG file...', end=' ', flush=True)
    dag_file = DAGFile()
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create a submit file for training and add it to DAG
    # -------------------------------------------------------------------------

    name = '01_train_models'

    # Create submit file and add job
    print(f'Creating {name}.sub...', end=' ', flush=True)
    submit_file = SubmitFile(
        clusterlogs_dir=clusterlogs_dir.as_posix(),
        memory=expected_job_memory,
        cpus=8,
    )
    submit_file.add_job(
        name=name,
        job_script=(experiment_dir / f'{name}.py').as_posix(),
        arguments={'roi-split': '$(Process)', 'n-roi-splits': str(n_splits)},
        bid=bid,
        queue=int(n_splits),
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
    # Create a submit file for merging the HDF results files and add it to DAG
    # -------------------------------------------------------------------------

    name = '02_merge_hdf_files'

    print(f'Creating {name}.sub...', end=' ', flush=True)
    submit_file = SubmitFile(
        clusterlogs_dir=clusterlogs_dir.as_posix(),
        memory=expected_total_memory,
        cpus=1,
    )
    submit_file.add_job(
        name=name,
        job_script=(experiment_dir / f'{name}.py').as_posix(),
        arguments=dict(),
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
    dag_file.add_dependency('01_train_models', name)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create a submit file for stage 2 (find hypothesis, compute MF, ...)
    # -------------------------------------------------------------------------

    name = '03_run_stage_2'

    # Create submit file and add job
    print(f'Creating {name}.sub...', end=' ', flush=True)
    submit_file = SubmitFile(
        clusterlogs_dir=clusterlogs_dir.as_posix(),
        memory=expected_total_memory,
        cpus=4,
    )
    submit_file.add_job(
        name=name,
        job_script=(experiment_dir / f'{name}.py').as_posix(),
        arguments=dict(),
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
    dag_file.add_dependency('02_merge_hdf_files', name)
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
