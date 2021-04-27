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

from hsr4hci.coordinates import get_center, cartesian2polar
from hsr4hci.data import load_parang, load_psf_template
from hsr4hci.forward_modeling import add_fake_planet
from hsr4hci.general import rotate_position
from hsr4hci.masking import get_exclusion_mask, get_selection_mask
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

    parang = load_parang(name=dataset)
    psf_template = load_psf_template(name=dataset)

    n_frames = len(parang)

    # -------------------------------------------------------------------------
    # Load the data set; set up units; define shortcuts
    # -------------------------------------------------------------------------

    signal_time = int(0.5 * (n_frames - 1))
    position = (42, 23)

    # Compute assumed planet path
    n = n_frames // 100
    parang_resampled = parang[::n]
    final_position = rotate_position(
        position=position,
        center=center,
        angle=float(parang[int(signal_time)]),
    )
    _, planet_positions = add_fake_planet(
        stack=np.zeros((len(parang_resampled), frame_size[0], frame_size[1])),
        parang=parang_resampled,
        psf_template=psf_template,
        polar_position=cartesian2polar(
            position=(final_position[0], final_position[1]),
            frame_size=frame_size,
        ),
        magnitude=1,
        extra_scaling=1,
        dit_stack=1,
        dit_psf_template=1,
        return_planet_positions=True,
    )
    planet_positions = np.asarray(planet_positions)

    # Set up a new figure
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')
    disable_ticks(ax)

    # Prepare grid for the pcolormesh()
    x_range = np.arange(frame_size[0])
    y_range = np.arange(frame_size[1])
    x, y = np.meshgrid(x_range, y_range)

    selection_mask = get_selection_mask(
        mask_size=frame_size,
        position=position,
        parang=parang,
        signal_time=signal_time,
        psf_template=psf_template,
        annulus_width=Quantity(0, 'pixel'),
        radius_position=Quantity(12, 'pixel'),
        radius_mirror_position=Quantity(12, 'pixel'),
    )
    ax.pcolormesh(
        x,
        y,
        selection_mask,
        cmap=get_transparent_cmap('C0'),
        shading='nearest',
    )

    exclusion_mask = get_exclusion_mask(
        mask_size=frame_size,
        position=position,
        parang=parang,
        signal_time=signal_time,
        psf_template=psf_template,
    )
    ax.pcolormesh(
        x,
        y,
        exclusion_mask,
        cmap=get_transparent_cmap('C1'),
        shading='nearest',
    )

    # Plot planet positions
    ax.plot(
        planet_positions[:, 0],
        planet_positions[:, 1],
        lw=3,
        color='gray',
        solid_capstyle='round',
    )

    # Plot the center of the frame and the target pixel
    plt.plot(center[0], center[1], '+k')
    plt.plot(
        position[0],
        position[1],
        ms=3,
        marker='s',
        markerfacecolor='white',
        markeredgecolor='black',
    )

    legend_elements = [
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
        Line2D(
            [],
            [],
            lw=3,
            color='gray',
            solid_capstyle='round',
            label='Assumed planet path',
        ),
        Patch(facecolor='C0', label='Used as predictors'),
        Patch(facecolor='C1', label='Exclusion region'),
    ]

    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        frameon=False,
        ncol=2,
        fontsize=8,
    )

    # Save plot to PDF
    fig.tight_layout()
    file_path = Path('.') / 'predictor_selection.pdf'
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0.025)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
