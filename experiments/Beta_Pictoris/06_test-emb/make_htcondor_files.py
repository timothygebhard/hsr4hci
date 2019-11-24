"""
Make submit files to train models in parallel on the cluster.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import os
import shutil
import time

from hsr4hci.utils.config import load_config


# -----------------------------------------------------------------------------
# AUXILIARY FUNCTIONS
# -----------------------------------------------------------------------------

def abs_join(*args):
    """Join *args and call abspath() on the result."""
    return os.path.abspath(os.path.join(*args))


def get_arguments() -> argparse.Namespace:
    """
    Parse and return the command line arguments.

    Returns:
        An `argparse.Namespace` object containing the command line
        options that were passed to this script.
    """

    # Set up a parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--n-jobs',
                        type=int,
                        metavar='N',
                        default=256,
                        help='Number of concurrent cluster jobs into which to '
                             'split the training. Default: 256.')
    parser.add_argument('--n-rounds',
                        type=int,
                        metavar='N',
                        default=10,
                        help='Number of rounds for the E/M algorithm. '
                             'Default: 10.')

    # Parse and return the command line arguments
    return parser.parse_args()


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nMAKE HTCONDOR FILES\n', flush=True)

    # -------------------------------------------------------------------------
    # Load config and define shortcuts
    # -------------------------------------------------------------------------

    # Get the command line arguments
    args = get_arguments()
    n_jobs = int(args.n_jobs)
    n_rounds = int(args.n_rounds)

    print('Arguments:', flush=True)
    print('Number of concurrent jobs (n_jobs):', n_jobs)
    print('Number of EM iterations (n_rounds):', n_rounds)
    print()

    # Load experiment config from JSON
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    config = load_config(os.path.join(experiment_dir, 'config.json'))

    # Define shortcuts
    frame_size = config['dataset']['frame_size']

    # -------------------------------------------------------------------------
    # Create directory for htcondor and clusterlogs
    # -------------------------------------------------------------------------

    # If the htcondor directory exists already, delete it (to get rid of all
    # old log files). Then, create a fresh version of it
    htcondor_dir = os.path.join(experiment_dir, 'htcondor')
    if os.path.isdir(htcondor_dir):
        print('Deleting existing htcondor directory...', end=' ', flush=True)
        shutil.rmtree(htcondor_dir, ignore_errors=True)
        print('Done!', flush=True)
    
    print('Creating htcondor directory...', end=' ', flush=True)
    Path(htcondor_dir).mkdir(exist_ok=True)
    print('Done!', flush=True)

    # Create dir for clusterlogs
    print('Creating clusterlogs directory...', end=' ', flush=True)
    clusterlogs_dir = os.path.join(htcondor_dir, 'clusterlogs')
    Path(clusterlogs_dir).mkdir(exist_ok=True)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create train.sub file
    # -------------------------------------------------------------------------

    print('Creating train.sub file...', end=' ', flush=True)

    # Path to script that will run the training
    train_script_path = \
        os.path.abspath(os.path.join(experiment_dir, 'train_emb.py'))

    # Add file header with requirements
    submit_file_lines = \
        ['#' + 78 * '-',
         '# GENERAL JOB REQUIREMENTS',
         '#' + 78 * '-',
         '',
         'executable = /is/cluster/tgebhard/.virtualenvs/hsr4hci/bin/python3',
         'getenv = True',
         '',
         'request_memory = 16384',
         'request_cpus = 1',
         '\n',
         '#' + 78 * '-',
         '# JOBS',
         '#' + 78 * '-']

    # Loop over all regions and add a job for each of them
    for region_idx in range(n_jobs):

        # Define paths for output, error and log file
        out_path = \
            abs_join(clusterlogs_dir, f"train_$(emb_round)_{region_idx}.out")
        err_path = \
            abs_join(clusterlogs_dir, f"train_$(emb_round)_{region_idx}.err")
        log_path = \
            abs_join(clusterlogs_dir, f"train_$(emb_round)_{region_idx}.log")

        # Gather arguments: script + command line arguments
        arguments = [train_script_path,
                     f'--n-regions {n_jobs}',
                     f'--region-idx {region_idx}',
                     f'--emb-round $(emb_round)']

        # Collect lines for the current region
        job_lines = \
            [f'\n# REGION {region_idx + 1} / {n_jobs}',
             f'output = {out_path}',
             f'error = {err_path}',
             f'log = {log_path}',
             f'arguments = {" ".join(arguments)}',
             'priority = -950',
             'queue']

        # Add lines to the submit file lines
        submit_file_lines += job_lines

    # Create and save the train.sub file
    file_path = os.path.join(htcondor_dir, 'train.sub')
    with open(file_path, 'w') as submit_file:
        submit_file.write('\n'.join(submit_file_lines))

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create merge.sub file
    # -------------------------------------------------------------------------

    print('Creating merge.sub file...', end=' ', flush=True)

    # Path to script that will merge the residuals
    merge_script_path = \
        os.path.abspath(os.path.join(experiment_dir, 'merge_residuals.py'))

    # Add file header with requirements
    submit_file_lines = \
        ['#' + 78 * '-',
         '# GENERAL JOB REQUIREMENTS',
         '#' + 78 * '-',
         '',
         'executable = /is/cluster/tgebhard/.virtualenvs/hsr4hci/bin/python3',
         'getenv = True',
         '',
         'request_memory = 8192',
         'request_cpus = 2',
         '\n',
         '#' + 78 * '-',
         '# JOB',
         '#' + 78 * '-',
         '\n']

    # Define paths for output, error and log file
    out_path = abs_join(clusterlogs_dir, f"merge_$(emb_round).out")
    err_path = abs_join(clusterlogs_dir, f"merge_$(emb_round).err")
    log_path = abs_join(clusterlogs_dir, f"merge_$(emb_round).log")

    # Gather arguments: script + command line arguments
    arguments = [merge_script_path,
                 f'--emb-round $(emb_round)']

    # Collect lines for the job
    submit_file_lines += \
        [f'output = {out_path}',
         f'error = {err_path}',
         f'log = {log_path}',
         f'arguments = {" ".join(arguments)}',
         'priority = -950',
         'queue']

    # Create and save the train.sub file
    file_path = os.path.join(htcondor_dir, 'merge.sub')
    with open(file_path, 'w') as submit_file:
        submit_file.write('\n'.join(submit_file_lines))

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Create run.dag file
    # -------------------------------------------------------------------------

    print('Creating run.dag file...', end=' ', flush=True)

    # Add all jobs to DAG file
    dag_file_lines = list()
    for i in range(n_rounds):
        dag_file_lines.append(f'JOB TRAIN{i} train.sub')
        dag_file_lines.append(f'JOB MERGE{i} merge.sub')

    # Add variables with the EMB round
    dag_file_lines.append('')
    for i in range(n_rounds):
        dag_file_lines.append(f'VARS TRAIN{i} emb_round="{i}"')
        dag_file_lines.append(f'VARS MERGE{i} emb_round="{i}"')

    # Add PARENT -> CHILD relations to specify dependencies
    dag_file_lines.append('')
    for i in range(n_rounds):
        dag_file_lines.append(f'PARENT TRAIN{i} CHILD MERGE{i}')
        if i + 1 < n_rounds:
            dag_file_lines.append(f'PARENT MERGE{i} CHILD TRAIN{i + 1}')

    # Create and save the run.dag file
    file_path = os.path.join(htcondor_dir, 'run.dag')
    with open(file_path, 'w') as dag_file:
        dag_file.write('\n'.join(dag_file_lines))

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
