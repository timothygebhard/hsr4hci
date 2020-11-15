"""
Create submit files to run experiment on an HTCondor-based cluster.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from sys import getsizeof
from pathlib import Path

import argparse
import time
import os

from astropy.units import Quantity

import numpy as np

from hsr4hci.utils.config import load_config
from hsr4hci.utils.data import load_data
from hsr4hci.utils.htcondor import SubmitFile, DAGFile
from hsr4hci.utils.masking import get_roi_mask
from hsr4hci.utils.units import set_units_for_instrument


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


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nTRAIN HALF-SIBLING REGRESSION MODELS\n', flush=True)

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
    stack, parang, psf_template, observing_conditions, metadata = load_data(
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
            getsizeof(stack),
            getsizeof(parang),
            getsizeof(psf_template),
            getsizeof(observing_conditions),
            getsizeof(metadata),
        ]
    )

    # Compute the size of a single stack-shaped variable
    array_memory = getsizeof(np.ones((n_frames, n_pixels_per_job)))

    # Count the number of stack-shaped arrays that we need to keep in memory
    n_arrays = 2 + 3 * (config['signal_masking']['n_signal_times'] + 1)

    # Compute the expected amount of memory that we need per job (in MB)
    expected_job_memory = 4 * dataset_memory + n_arrays * array_memory
    expected_job_memory /= 1024 ** 2
    expected_job_memory *= 2
    expected_job_memory = int(expected_job_memory)

    # Compute the expected total memory needed for merging the HDF files
    expected_total_memory = n_arrays * getsizeof(stack)
    expected_total_memory /= 1024**2
    expected_total_memory *= 4
    expected_total_memory = int(expected_total_memory)

    print('')
    print(f'Pixels per job: {np.sum(roi_mask)} / {n_splits} <= '
          f'{n_pixels_per_job}')
    print('')
    print(f'Data set memory: {int(dataset_memory / 1024 ** 2):6d} MB')
    print(f'Expected memory: {expected_job_memory:6d} MB per job\n'
          f'                 {expected_total_memory:6d} MB in total')
    print('')

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
        name='train',
        job_script=(experiment_dir / f'{name}.py').as_posix(),
        arguments={'split': '$(Process)', 'n-splits': str(n_splits)},
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
    dag_file.add_dependency('01_train_models', name)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create a submit file for finding hypotheses and add it to DAG
    # -------------------------------------------------------------------------

    name = '03_find_hypotheses'

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
    # Create a submit file for computing the match fraction and add it to DAG
    # -------------------------------------------------------------------------

    name = '04_compute_matches'

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
    dag_file.add_dependency('03_find_hypotheses', name)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create a submit file for creating the signal estimates and add it to DAG
    # -------------------------------------------------------------------------

    name = '05_construct_signal_estimates'

    # Create submit file and add job
    print(f'Creating {name}.sub...', end=' ', flush=True)
    submit_file = SubmitFile(
        clusterlogs_dir=clusterlogs_dir.as_posix(), memory=16384, cpus=4
    )
    submit_file.add_job(
        name=name,
        job_script=(experiment_dir / f'{name}.py').as_posix(),
        arguments={},
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
    dag_file.add_dependency('04_compute_matches', name)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create a submit file for computing the SNR and add it to DAG
    # -------------------------------------------------------------------------

    name = '06_compute_snr'

    # Create submit file and add job
    print(f'Creating {name}.sub...', end=' ', flush=True)
    submit_file = SubmitFile(
        clusterlogs_dir=clusterlogs_dir.as_posix(), memory=8192, cpus=4
    )
    submit_file.add_job(
        name=name,
        job_script=(experiment_dir / f'{name}.py').as_posix(),
        arguments={},
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
    dag_file.add_dependency('05_construct_signal_estimates', name)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create a submit file for computing the SNR and add it to DAG
    # -------------------------------------------------------------------------

    name = '07_compute_roc_curve'

    # Create submit file and add job
    print(f'Creating {name}.sub...', end=' ', flush=True)
    submit_file = SubmitFile(
        clusterlogs_dir=clusterlogs_dir.as_posix(), memory=8192, cpus=4
    )
    submit_file.add_job(
        name=name,
        job_script=(experiment_dir / f'{name}.py').as_posix(),
        arguments={},
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
    dag_file.add_dependency('04_compute_matches', name)
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
