"""
Make submit files to train models in parallel on the cluster.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import os
import time

from hsr4hci.utils.config import load_config
from hsr4hci.utils.masking import get_positions_from_mask
from hsr4hci.utils.roi_selection import get_roi_mask


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
    print('\nMAKE SUBMIT FILE\n', flush=True)

    # -------------------------------------------------------------------------
    # Load config and define shortcuts
    # -------------------------------------------------------------------------

    # Load experiment config from JSON
    config = load_config('./config.json')

    # Define shortcuts
    frame_size = config['dataset']['frame_size']
    pixscale = config['dataset']['pixscale']
    roi_ier = config['experiment']['roi']['inner_exclusion_radius']
    roi_oer = config['experiment']['roi']['outer_exclusion_radius']
    experiment_dir = os.path.abspath(config['experiment_dir'])
    train_script_path = \
        os.path.abspath(os.path.join(experiment_dir, 'train_position.py'))

    # -------------------------------------------------------------------------
    # Create directory for clusterlogs
    # -------------------------------------------------------------------------

    clusterlogs_dir = \
        os.path.abspath(os.path.join(experiment_dir, 'clusterlogs'))
    Path(clusterlogs_dir).mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Get positions in ROI
    # -------------------------------------------------------------------------

    # Get ROI mask
    roi_mask = get_roi_mask(mask_size=frame_size,
                            pixscale=pixscale,
                            inner_exclusion_radius=roi_ier,
                            outer_exclusion_radius=roi_oer)

    # Get position in ROI
    roi_positions = get_positions_from_mask(roi_mask)

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

    for (x, y) in roi_positions:

        job_lines = \
            [f'\n# POSITION ({x}, {y})',
             f'output = {abs_join(clusterlogs_dir, f"{x}_{y}.out")}',
             f'error = {abs_join(clusterlogs_dir, f"{x}_{y}.err")}',
             f'log = {abs_join(clusterlogs_dir, f"{x}_{y}.log")}',
             f'arguments = {train_script_path} --x {x} --y {y}',
             'queue']

        submit_file_lines += job_lines

    # -------------------------------------------------------------------------
    # Create the submit file
    # -------------------------------------------------------------------------

    with open('submit.sub', 'w') as submit_file:
        submit_file.write('\n'.join(submit_file_lines))

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
