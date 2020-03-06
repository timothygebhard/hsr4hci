"""
Make submit files to train model on the cluster.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

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

    # Load experiment config from JSON
    experiment_dir = os.path.dirname(os.path.realpath(__file__))
    config = load_config(os.path.join(experiment_dir, 'config.json'))

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
        os.path.abspath(os.path.join(experiment_dir, 'train.py'))

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
         'request_cpus = 4',
         '\n',
         ''
         '#' + 78 * '-',
         '# RUNTIME LIMITATION AND AUTOMATIC RETRIES',
         '#' + 78 * '-',
         '',
         f'MaxRuntime = 86400',
         f'NumRetries = 0',
         '',
         'job_machine_attrs = Machine',
         'job_machine_attrs_history_length = 4',
         'requirements      = (target.machine =!= MachineAttrMachine1) && \\ ',
         '                    (target.machine =!= MachineAttrMachine2) && \\ ',
         '                    (target.machine =!= MachineAttrMachine3)',
         '',
         'periodic_hold         = (JobStatus == 2) && \\ ',
         '                        ((CurrentTime - EnteredCurrentStatus) >= '
         '$(MaxRuntime))',
         'periodic_hold_subcode = 1',
         'periodic_release      = (HoldReasonCode == 3) && \\ ',
         '                        (HoldReasonSubCode == 1) && \\ ',
         '                        (JobRunCount < $(NumRetries))',
         'periodic_hold_reason  = ifthenelse(JobRunCount < $(NumRetries), \\',
         '                                   "Maximum runtime exceeded!", \\ ',
         '                                   "No more retries left!")',
         '\n',
         '#' + 78 * '-',
         '# JOBS',
         '#' + 78 * '-',
         '']

    # Define paths for output, error and log file
    out_path = abs_join(clusterlogs_dir, f"train.out")
    err_path = abs_join(clusterlogs_dir, f"train.err")
    log_path = abs_join(clusterlogs_dir, f"train.log")

    # Collect lines for the current region
    job_lines = \
        [f'output = {out_path}',
         f'error  = {err_path}',
         f'log    = {log_path}',
         f'arguments = {train_script_path}',
         'priority  = -950',
         f'queue']

    # Add lines to the submit file lines
    submit_file_lines += job_lines

    # Create and save the train.sub file
    file_path = os.path.join(htcondor_dir, 'train.sub')
    with open(file_path, 'w') as submit_file:
        submit_file.write('\n'.join(submit_file_lines))

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
