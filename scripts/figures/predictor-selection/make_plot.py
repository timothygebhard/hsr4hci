"""
Plot the exclusion and selection mask for demonstration purposes.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import time

from astropy.units import Quantity
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import matplotlib.pyplot as plt
import numpy as np

from hsr4hci.coordinates import get_center
from hsr4hci.masking import (
    get_exclusion_mask,
    get_predictor_pixel_selection_mask,
)
from hsr4hci.plotting import get_transparent_cmap, disable_ticks


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
    # Load the data set; set up units; define shortcuts
    # -------------------------------------------------------------------------

    dataset = 'beta_pictoris__lp'
    frame_size = (65, 65)
    center = get_center(frame_size)

    position = (45, 16)

    # -------------------------------------------------------------------------
    # Load the data set; set up units; define shortcuts
    # -------------------------------------------------------------------------

    # Set up a new figure and adjust margins
    fig, axes = plt.subplots(
        ncols=2, figsize=(18.4 / 2.54 / 2, 18.4 / 2.54 / 4)
    )
    fig.subplots_adjust(
        left=0.0025, bottom=0.0025, top=0.9975, right=0.9975, wspace=0
    )

    # Disable ticks, fix aspect ratio
    for ax in axes:
        ax.set_aspect('equal')
        disable_ticks(ax)
    axes[1].axis('off')

    # Prepare grid for the pcolormesh()
    x_range = np.arange(frame_size[0])
    y_range = np.arange(frame_size[1])
    x, y = np.meshgrid(x_range, y_range)

    # Get the predictor pixel selection mask
    selection_mask = get_predictor_pixel_selection_mask(
        mask_size=frame_size,
        position=position,
        radius_position=Quantity(16, 'pixel'),
        radius_opposite=Quantity(16, 'pixel'),
        radius_excluded=Quantity(9, 'pixel'),
    )

    # Plot the predictor pixel selection mask
    axes[0].pcolormesh(
        x,
        y,
        selection_mask,
        cmap=get_transparent_cmap('C0'),
        shading='nearest',
    )

    # Get the exclusion mask
    exclusion_mask = get_exclusion_mask(
        mask_size=frame_size,
        position=position,
        radius_excluded=Quantity(9, 'pixel'),
    )

    # Plot the exclusion mask
    axes[0].pcolormesh(
        x,
        y,
        exclusion_mask,
        cmap=get_transparent_cmap('C1'),
        shading='nearest',
    )

    # Plot the center of the frame
    axes[0].plot(center[0], center[1], '*', color='gold', mew=1.5, ms=6)

    # Plot the target pixel
    axes[0].plot(
        position[0],
        position[1],
        ms=3,
        marker='s',
        markerfacecolor='white',
        markeredgecolor='black',
    )

    # -------------------------------------------------------------------------
    # Add custom legend to the plot
    # -------------------------------------------------------------------------

    handles = [
        Line2D(
            [0],
            [0],
            marker='*',
            ls='',
            color='gold',
            mew=1.5,
            ms=6,
            label='Position of the host star',
        ),
        Line2D(
            [],
            [],
            ls='None',
            label='Target pixel',
            ms=3,
            marker='s',
            markerfacecolor='white',
            markeredgecolor='black',
        ),
        Patch(facecolor='C0', label='Pixels used as predictors'),
        Patch(facecolor='C1', label='Exclusion region'),
    ]

    axes[1].legend(
        handles=handles,
        loc='center left',
        borderaxespad=1,
        borderpad=0,
        frameon=False,
        fontsize=6,
    )

    # Save plot to PDF
    file_path = Path('.') / 'predictor_selection.pdf'
    plt.savefig(file_path, pad_inches=0.0)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
