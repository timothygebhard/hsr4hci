"""
Make submit files to train models in parallel on the cluster.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import os
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
    print('\nMAKE SUBMIT FILE\n', flush=True)

    # -------------------------------------------------------------------------
    # Load config and define shortcuts
    # -------------------------------------------------------------------------

    # Get the command line arguments
    args = get_arguments()
    n_jobs = int(args.n_jobs)

    # Load experiment config from JSON
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    config = load_config(os.path.join(experiment_dir, 'config.json'))

    # Define shortcuts
    frame_size = config['dataset']['frame_size']
    pixscale = config['dataset']['pixscale']
    roi_ier = config['experiment']['roi']['inner_exclusion_radius']
    roi_oer = config['experiment']['roi']['outer_exclusion_radius']
    train_script_path = \
        os.path.abspath(os.path.join(experiment_dir, 'train_region.py'))

    # -------------------------------------------------------------------------
    # Create directory for clusterlogs
    # -------------------------------------------------------------------------

    clusterlogs_dir = os.path.join(experiment_dir, 'clusterlogs')
    Path(clusterlogs_dir).mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Create header of submit file
    # -------------------------------------------------------------------------

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
         '# JOBS',
         '#' + 78 * '-']

    # -------------------------------------------------------------------------
    # Loop over all positions in the ROI and add a job
    # -------------------------------------------------------------------------

    for region_idx in range(n_jobs):

        job_lines = \
            [f'\n# REGION {region_idx + 1} / {n_jobs}',
             f'output = {abs_join(clusterlogs_dir, f"{region_idx}.out")}',
             f'error = {abs_join(clusterlogs_dir, f"{region_idx}.err")}',
             f'log = {abs_join(clusterlogs_dir, f"{region_idx}.log")}',
             f'arguments = {train_script_path} --n-regions {n_jobs} '
             f'--region-idx {region_idx}',
             'queue']

        submit_file_lines += job_lines

    # -------------------------------------------------------------------------
    # Create the submit file
    # -------------------------------------------------------------------------

    file_path = os.path.join(experiment_dir, 'submit.sub')
    with open(file_path, 'w') as submit_file:
        submit_file.write('\n'.join(submit_file_lines))

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
