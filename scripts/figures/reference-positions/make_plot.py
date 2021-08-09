"""
Plot (reference) positions for SNR/FPF computation.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import time

from astropy.units import Quantity
from photutils import CircularAperture
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import numpy as np

from hsr4hci.coordinates import get_center, polar2cartesian
from hsr4hci.plotting import disable_ticks
from hsr4hci.positions import (
    get_reference_positions,
    rotate_reference_positions,
)


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nPLOT REFERENCE POSITIONS\n', flush=True)

    # -------------------------------------------------------------------------
    # Define frame size and planet position; compute reference positions
    # -------------------------------------------------------------------------

    # Define a frame size and compute the frame center
    frame_size = (41, 41)
    center = get_center(frame_size)

    # Define a FWHM for the PSF (choose this much larger than "normal" so that
    # you can actually see the different "apertures" in the plot)
    psf_fwhm = 12

    # Define the position of the planet candidate (polar and Cartesian)
    polar_position = (Quantity(16, 'pixel'), Quantity(270, 'degree'))
    cartesian_position = polar2cartesian(
        *polar_position, frame_size=frame_size
    )

    # Compute the reference positions for the planet candidate position
    reference_positions = get_reference_positions(
        polar_position=polar_position,
        aperture_radius=Quantity(psf_fwhm / 2, 'pixel'),
        exclusion_angle=Quantity(0, 'degree'),
    )

    # Compute the rotated reference positions
    rotated_reference_positions = rotate_reference_positions(
        reference_positions[1:-1], 2
    )

    # -------------------------------------------------------------------------
    # Set up the figure and plot the planet candidate position
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

    # Create an empty imshow plot with the right size; this allows us to use
    # the 2D coordinate system that is implicitly defined by it
    ax = axes[0]
    ax.imshow(np.zeros(frame_size), origin='lower', cmap='Greys')
    ax.plot(center[0], center[1], '*', color='gold', mew=1.5, ms=5)

    # Plot the position of the planet candidate
    ax.plot(cartesian_position[0], cartesian_position[1], 'xk', mew=1.5, ms=5)
    CircularAperture(cartesian_position, r=(psf_fwhm / 2)).plot(
        **dict(axes=ax, ls='--', lw=0.5)
    )

    # -------------------------------------------------------------------------
    # Plot the (rotated) reference positions
    # -------------------------------------------------------------------------

    # Plot the positions that we exclude due to potential bias
    for position in (reference_positions[0], reference_positions[-1]):
        CircularAperture(
            polar2cartesian(*position, frame_size=frame_size), r=(psf_fwhm / 2)
        ).plot(**dict(color='red', hatch='///////', ls='--', axes=ax, lw=0.5))

    # Plot the rotated reference positions
    for i, (rotated, ls) in enumerate(
        zip(rotated_reference_positions, ('--', ':', '-.'))
    ):
        for _ in rotated:
            pos = polar2cartesian(*_, frame_size=frame_size)
            ax.plot(pos[0], pos[1], f'{i + 1}', color=f'C{i}', mew=1.5, ms=5)
            CircularAperture(pos, r=(psf_fwhm / 2)).plot(
                **dict(color=f'C{i}', axes=ax, ls=ls, lw=0.5)
            )

    # -------------------------------------------------------------------------
    # Add a legend to the plot
    # -------------------------------------------------------------------------

    # Define default options for plotting
    kwargs = dict(ls='', mew=1.5, ms=5)

    # Define handles
    handles = [
        Line2D(
            [0],
            [0],
            marker='*',
            color='gold',
            label='Position of the host star',
            **kwargs,
        ),
        Line2D(
            [0],
            [0],
            marker='x',
            color='k',
            label='Position of planet candidate',
            **kwargs,
        ),
        Patch(
            facecolor='none',
            edgecolor='r',
            lw=0.5,
            ls='--',
            hatch='///////',
            label='Excluded due to potential bias',
        ),
        Patch(facecolor='none', edgecolor='none', label=''),
        Line2D(
            [0],
            [0],
            marker='1',
            color='C0',
            label='Reference positions (set #1)',
            **kwargs,
        ),
        Line2D(
            [0],
            [0],
            marker='2',
            color='C1',
            label='Reference positions (set #2)',
            **kwargs,
        ),
        Line2D(
            [0],
            [0],
            marker='3',
            color='C2',
            label='Reference positions (set #3)',
            **kwargs,
        ),
    ]

    # Add the legend to the plot
    axes[1].legend(
        handles=handles,
        loc='center left',
        borderaxespad=1,
        borderpad=0,
        frameon=False,
        fontsize=6,
    )

    # Save plot to PDF
    file_path = Path('.') / 'reference-positions.pdf'
    plt.savefig(file_path, pad_inches=0.0)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
