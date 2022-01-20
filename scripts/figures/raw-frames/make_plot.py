"""
Plot raw frames for all data sets for illustrative purposes.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import time

import numpy as np

from hsr4hci.data import load_metadata, load_stack
from hsr4hci.plotting import plot_frame


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nMAKE PLOT\n', flush=True)

    # -------------------------------------------------------------------------
    # Loop over all data sets and create a plot of the first frame
    # -------------------------------------------------------------------------

    for dataset in (
        'beta_pictoris__lp',
        'beta_pictoris__mp',
        'hr_8799__lp',
        'r_cra__lp',
    ):

        print(f'Plotting {dataset}...', end=' ', flush=True)

        # Load stack and metadata
        stack = load_stack(name_or_path=dataset, frame_size=(101, 101))
        metadata = load_metadata(name_or_path=dataset)

        # Select the frame for the plot
        frame = stack[0]
        vmin = np.min(frame)
        vmax = np.max(frame)

        # Create the plot
        file_path = Path('.') / f'{dataset}.pdf'
        plot_frame(
            frame=frame,
            positions=[],
            labels=[],
            pixscale=metadata['PIXSCALE'],
            figsize=(4.0 / 2.54, 4.0 / 2.54),
            subplots_adjust=dict(
                left=0.005, right=0.995, bottom=0.005, top=0.995,
            ),
            aperture_radius=0,
            label_positions=None,
            scalebar_color='white',
            cmap='viridis',
            limits=(vmin, vmax),
            use_logscale=True,
            add_colorbar=False,
            add_scalebar=True,
            add_cardinal_directions=False,
            scalebar_loc='upper right',
            file_path=file_path,
        )

        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
