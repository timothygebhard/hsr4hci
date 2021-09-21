"""
Plot the results of stage 2: hypothesis match, match fraction map,
polar match fraction map, cross-correlation between the polar match
fraction map and the expected signal template, and the final residual
selection mask.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from math import fmod
from pathlib import Path
from typing import List, Tuple

import argparse
import os
import time

from astropy.units import Quantity
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage.filters import gaussian
from skimage.measure import find_contours

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from hsr4hci.config import load_config
from hsr4hci.coordinates import get_center
from hsr4hci.data import (
    load_metadata,
    load_parang,
    load_planets,
    load_psf_template,
)
from hsr4hci.fits import read_fits
from hsr4hci.forward_modeling import add_fake_planet
from hsr4hci.general import crop_center
from hsr4hci.masking import get_roi_mask
from hsr4hci.residuals import get_residual_selection_mask
from hsr4hci.plotting import (
    disable_ticks,
    add_colorbar_to_ax,
    get_cmap,
    set_fontsize,
)
from hsr4hci.units import InstrumentUnitsContext


# -----------------------------------------------------------------------------
# DEFINE AUXILIARY FUNCTIONS
# -----------------------------------------------------------------------------

def _prepare_plot__dec_rac(
    frame_size: Tuple[int, int],
    contours: List[np.ndarray],
    pixscale: float,
) -> Tuple[Figure, Axes]:
    """
    Auxiliary function to prepare a plot in a declination / right
    ascension coordinate system.
    """

    # Create new figure
    fig, ax = plt.subplots(figsize=(3.4 / 2.54, 4.2 / 2.54))

    # Set various plot options
    ax.set_aspect('equal')
    disable_ticks(ax)
    set_fontsize(ax=ax, fontsize=6)

    # Add axis labels
    ax.set_xlabel('Right Ascension', labelpad=2)
    ax.set_ylabel('Declination', labelpad=2)

    # Add a scale bar
    scalebar = AnchoredSizeBar(
        transform=ax.transData,
        size=0.3 / pixscale,
        label='0.3"',
        loc=2,
        pad=0.5,
        color='white',
        frameon=False,
        size_vertical=0,
        fontproperties=fm.FontProperties(size=6),
    )
    ax.add_artist(scalebar)

    # Add a + at the location of the center
    center = get_center(frame_size)
    ax.plot(center[0], center[1], '+', ms=5, mew=1, color='white', zorder=99)

    # Add contour for the true planet path
    for contour in contours:
        ax.plot(
            contour[:, 1],
            contour[:, 0],
            color='white',
            lw=1,
            solid_capstyle='round',
        )

    return fig, ax


def _prepare_plot__sep_ang() -> Tuple[Figure, Axes]:
    """
    Auxiliary function to prepare a plot in a polar coordinate system
    given by the separation and the azimuthal angle.
    """

    # Create new figure
    fig, ax = plt.subplots(figsize=(3.4 / 2.54, 4.2 / 2.54))

    # Set various plot options
    ax.set_aspect('equal')
    disable_ticks(ax)
    set_fontsize(ax=ax, fontsize=6)

    # Add axis labels
    ax.set_xlabel('Azimuthal angle', labelpad=2)
    ax.set_ylabel('Separation', labelpad=2)

    return fig, ax


def plot_hypothesis_map(
    hypotheses: np.ndarray,
    contours: List[np.ndarray],
    plots_dir: Path,
    pixscale: float,
    x_lim: Tuple[float, float],
    y_lim: Tuple[float, float],
) -> None:
    """
    Create plot of the hypothesis map.
    """

    print('Plotting hypothesis map...', end=' ', flush=True)

    # Define shortcuts
    frame_size = (hypotheses.shape[0], hypotheses.shape[1])

    # Prepare a figure and adjust the margins
    fig, ax = _prepare_plot__dec_rac(
        frame_size=frame_size, contours=contours, pixscale=pixscale
    )
    fig.subplots_adjust(left=0.085, bottom=-0.055, right=0.96, top=0.96)

    # Plot match fraction map
    img = ax.pcolormesh(
        *np.meshgrid(np.arange(frame_size[0]), np.arange(frame_size[1])),
        hypotheses,
        vmin=0.0,
        vmax=1.0,
        shading='nearest',
        cmap=get_cmap('viridis'),
        snap=True,
        rasterized=True,
    )

    # Set the plot limits
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)

    # Add a colorbar and set options
    cbar = add_colorbar_to_ax(img, fig, ax, where='top')
    cbar.ax.tick_params(labelsize=5, pad=0.5, length=2)
    cbar.set_ticks(np.linspace(0, 1, 3))
    cbar.set_label(label='Relative temporal index', fontsize=6)

    # Save the results
    file_path = plots_dir / 'hypotheses.pdf'
    plt.savefig(file_path, pad_inches=0, dpi=600)
    plt.close()

    print('Done!', flush=True)


def plot_match_fraction_map(
    match_fraction: np.ndarray,
    contours: List[np.ndarray],
    plots_dir: Path,
    pixscale: float,
    x_lim: Tuple[float, float],
    y_lim: Tuple[float, float],
) -> None:
    """
    Create plot of the (Cartesian) match fraction map.
    """

    print('Plotting match fraction map...', end=' ', flush=True)

    # Define shortcuts
    frame_size = (match_fraction.shape[0], match_fraction.shape[1])

    # Prepare a figure and adjust the margins
    fig, ax = _prepare_plot__dec_rac(
        frame_size=frame_size, contours=contours, pixscale=pixscale
    )
    fig.subplots_adjust(left=0.085, bottom=-0.055, right=0.96, top=0.96)

    # Plot match fraction map
    img = ax.pcolormesh(
        *np.meshgrid(np.arange(frame_size[0]), np.arange(frame_size[1])),
        match_fraction,
        vmin=0.0,
        vmax=0.5,
        shading='nearest',
        cmap=get_cmap('viridis'),
        snap=True,
        rasterized=True,
    )

    # Set the plot limits
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)

    # Add a colorbar and set options
    cbar = add_colorbar_to_ax(img, fig, ax, where='top')
    cbar.ax.tick_params(labelsize=5, pad=0.5, length=2)
    cbar.set_ticks(
        np.linspace(0, 0.5, 6),
    )
    cbar.set_label(label='Match fraction', fontsize=6)

    # Save the results
    file_path = plots_dir / 'match_fraction.pdf'
    plt.savefig(file_path, pad_inches=0, dpi=600)
    plt.close()

    print('Done!', flush=True)


def plot_selection_mask(
    selection_mask: np.ndarray,
    contours: List[np.ndarray],
    plots_dir: Path,
    pixscale: float,
    x_lim: Tuple[float, float],
    y_lim: Tuple[float, float],
) -> None:
    """
    Create plot of the residual selection mask.
    """

    print('Plotting selection mask...', end=' ', flush=True)

    # Define shortcuts
    frame_size = (selection_mask.shape[0], selection_mask.shape[1])

    # Prepare a figure and adjust the margins
    fig, ax = _prepare_plot__dec_rac(
        frame_size=frame_size, contours=contours, pixscale=pixscale
    )
    # fig.subplots_adjust(left=0.085, bottom=-0.055, right=0.96, top=0.96)
    fig.subplots_adjust(left=0.085, bottom=-0.12075, right=0.96, top=0.96)

    # Plot match fraction map
    ax.pcolormesh(
        *np.meshgrid(np.arange(frame_size[0]), np.arange(frame_size[1])),
        selection_mask,
        vmin=0.0,
        vmax=1.0,
        shading='nearest',
        cmap=get_cmap('viridis'),
        snap=True,
        rasterized=True,
    )

    # Set the plot limits
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)

    # Add a custom legend
    ax.legend(
        handles=[
            Patch(
                facecolor=(68 / 255, 1 / 255, 84 / 255),
                edgecolor=None,
                label='Use default model',
            ),
            Patch(
                facecolor=(253 / 255, 231 / 255, 37 / 255),
                edgecolor=None,
                label='Use hypothesis model',
            ),
        ],
        bbox_to_anchor=(0, 1.07, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
        borderpad=0,
        ncol=1,
        fontsize=6,
        frameon=False,
    )

    # Save the results
    file_path = plots_dir / 'selection_mask.pdf'
    plt.savefig(file_path, pad_inches=0, dpi=600)
    plt.close()

    print('Done!', flush=True)


def plot_polar_match_fraction(
    polar_match_fraction: np.ndarray,
    expected_signal: np.ndarray,
    plots_dir: Path,
) -> None:
    """
    Create plot of the (polar) match fraction map.
    """

    print('Plotting polar match fraction map...', end=' ', flush=True)

    # Define shortcuts
    frame_size = (polar_match_fraction.shape[0], polar_match_fraction.shape[1])

    # Prepare a figure and adjust the margins
    fig, ax = _prepare_plot__sep_ang()
    fig.subplots_adjust(left=0.085, bottom=-0.055, right=0.96, top=0.96)

    # Plot match fraction map
    img = ax.pcolormesh(
        *np.meshgrid(np.arange(frame_size[0]), np.arange(frame_size[1])),
        polar_match_fraction,
        vmin=0.0,
        vmax=1.0,
        shading='nearest',
        cmap=get_cmap('viridis'),
        snap=True,
        rasterized=True,
    )

    # Add inset axis for signal template
    inset_ax = inset_axes(ax, width="33.3%", height="33.3%", loc=1)
    inset_ax.set_aspect('equal')
    disable_ticks(inset_ax)

    # Plot template for matching in upper right-hand corner
    inset_ax.pcolormesh(
        *np.meshgrid(
            np.arange(expected_signal.shape[0]),
            np.arange(expected_signal.shape[1]),
        ),
        expected_signal,
        vmin=0,
        vmax=1,
        shading='nearest',
        cmap='viridis',
        rasterized=True,
    )

    # Change border color of the inset axis
    for position in ('top', 'bottom', 'left', 'right'):
        inset_ax.spines[position].set_color('white')

    # Add a colorbar and set options
    cbar = add_colorbar_to_ax(img, fig, ax, where='top')
    cbar.ax.tick_params(labelsize=5, pad=0.5, length=2)
    cbar.set_ticks(np.linspace(0, 1, 3))
    cbar.set_label(label='Rescaled match fraction', fontsize=6)

    # Save the results
    file_path = plots_dir / 'polar_match_fraction.pdf'
    plt.savefig(file_path, pad_inches=0, dpi=600)
    plt.close()

    print('Done!', flush=True)


def plot_template_matching(
    matched: np.ndarray,
    plots_dir: Path,
    dec_rac_center: Tuple[float, float],
) -> None:
    """
    Create plot with the results of the cross-correlation.
    """

    print('Plotting cross-correlation results...', end=' ', flush=True)

    # Define shortcuts
    frame_size = (matched.shape[0], matched.shape[1])

    # Prepare a figure and adjust the margins
    fig, ax = _prepare_plot__sep_ang()
    fig.subplots_adjust(left=0.085, bottom=-0.055, right=0.96, top=0.96)

    # Plot the result cross-correlation the polar MF with the expected signal
    img = ax.pcolormesh(
        *np.meshgrid(np.arange(frame_size[0]), np.arange(frame_size[1])),
        matched,
        shading='nearest',
        cmap=get_cmap('viridis'),
        rasterized=True,
        vmin=0,
        vmax=1,
    )
    ax.set_aspect('equal')

    # Plot the peaks that we have found in the cross-correlation results
    for peak_position in peak_positions:
        x = (
            fmod(peak_position[1] + np.pi, 2 * np.pi)
            / (2 * np.pi)
            * matched.shape[1]
        )
        y = (
            peak_position[0]
            / min(dec_rac_center[0], dec_rac_center[1])
            * matched.shape[0]
        )
        plt.plot(x, y, 'xk', ms=3)

    # Add a colorbar and set options
    cbar = add_colorbar_to_ax(img, fig, ax, where='top')
    cbar.ax.tick_params(labelsize=5, pad=0.5, length=2)
    cbar.set_ticks(np.linspace(0, 1, 3))
    cbar.set_label(label='Cross-correlation value', fontsize=6)

    # Save the results
    file_path = plots_dir / 'template_matching.pdf'
    plt.savefig(file_path, pad_inches=0, dpi=600)
    plt.close()

    print('Done!', flush=True)


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nPLOT RESULTS OF STAGE 2\n', flush=True)

    # -------------------------------------------------------------------------
    # Set up parser to get command line arguments
    # -------------------------------------------------------------------------

    # Set up the parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-dir',
        type=str,
        required=True,
        metavar='PATH',
        help='(Absolute) path to experiment directory.',
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load experiment configuration and data
    # -------------------------------------------------------------------------

    # Get experiment directory
    experiment_dir = Path(os.path.expanduser(args.experiment_dir))
    if not experiment_dir.exists():
        raise NotADirectoryError(f'{experiment_dir} does not exist!')

    # Prepare the plots directory (where we will save the results)
    plots_dir = experiment_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    config = load_config(experiment_dir / 'config.json')
    print('Done!', flush=True)

    # Load the data set (parallactic angles, PSF template, metadata, planets)
    print('Loading data set...', end=' ', flush=True)
    parang = load_parang(**config['dataset'])
    psf_template = load_psf_template(**config['dataset'])
    metadata = load_metadata(**config['dataset'])
    planets = load_planets(**config['dataset'])
    print('Done!', flush=True)

    # Define the unit conversion context for this data set
    pixscale = metadata['PIXSCALE']
    lambda_over_d = metadata['LAMBDA_OVER_D']
    instrument_unit_context = InstrumentUnitsContext(
        pixscale=Quantity(pixscale, 'arcsec / pix'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
    )

    # Define quantities related to the size of the data set
    n_frames = len(parang)
    frame_size = (
        int(config['dataset']['frame_size'][0]),
        int(config['dataset']['frame_size'][1]),
    )
    dec_rac_center = get_center(frame_size)

    # Construct the mask for the region of interest (ROI)
    with instrument_unit_context:
        roi_mask = get_roi_mask(
            mask_size=frame_size,
            inner_radius=Quantity(*config['roi_mask']['inner_radius']),
            outer_radius=Quantity(*config['roi_mask']['outer_radius']),
        )

    # -------------------------------------------------------------------------
    # Compute a mask for the pixels that we know should contain planet signal
    # -------------------------------------------------------------------------

    print('Determining affected pixels...', end=' ', flush=True)

    # Normalize the PSF template
    psf_template /= np.max(psf_template)

    # Clip the PSF template for determination of affected pixels
    clipped_psf_template = np.copy(psf_template)
    clipped_psf_template[psf_template < 0.2] = 0

    # Down-sample the parallactic angle (to speed up the computation)
    n = n_frames // 10
    parang_resampled = parang[::n]

    # Initialize the hypothesized stack and mask of affected pixels
    hypothesized_stack = np.zeros((len(parang_resampled),) + frame_size)

    # Loop over (potentially multiple) planets and compute their stack
    with instrument_unit_context:
        for name, parameters in planets.items():
            signal_stack = np.array(
                add_fake_planet(
                    stack=hypothesized_stack,
                    parang=parang_resampled,
                    psf_template=clipped_psf_template,
                    polar_position=(
                        Quantity(parameters['separation'], 'arcsec'),
                        Quantity(parameters['position_angle'], 'degree'),
                    ),
                    magnitude=0,
                    extra_scaling=1,
                    dit_stack=1,
                    dit_psf_template=1,
                    return_planet_positions=False,
                    interpolation='bilinear',
                )
            )
            signal_stack /= np.max(signal_stack)
            hypothesized_stack += signal_stack

    # Determine the mask with all pixels affected by planets
    affected_mask = np.max(hypothesized_stack, axis=0) > 0.1
    affected_mask = gaussian(affected_mask, sigma=2)
    contours = find_contours(affected_mask, 0.5)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Load hypotheses map / match fraction, compute selection mask etc.
    # -------------------------------------------------------------------------

    # Load the hypotheses from FITS
    print('Loading hypothesis map...', end=' ', flush=True)
    file_path = experiment_dir / 'hypotheses' / 'hypotheses.fits'
    hypotheses = read_fits(file_path, return_header=False)
    hypotheses /= n_frames
    hypotheses[~roi_mask] = np.nan
    print('Done!', flush=True)

    # Load the match fraction from FITS
    print('Loading match fraction map...', end=' ', flush=True)
    file_path = experiment_dir / 'match_fractions' / 'median_mf.fits'
    match_fraction = read_fits(file_path, return_header=False)
    match_fraction[~roi_mask] = np.nan
    print('Done!', flush=True)

    # Compute the selection mask (and intermediate quantities)
    print('Computing selection mask...', end=' ', flush=True)
    (
        selection_mask,
        polar_match_fraction,
        matched,
        expected_signal,
        peak_positions,
    ) = get_residual_selection_mask(
        match_fraction=match_fraction,
        parang=parang,
        psf_template=psf_template,
    )
    selection_mask = selection_mask.astype(float)
    selection_mask[~roi_mask] = np.nan
    print('Done!\n', flush=True)

    # Crop expected signal
    expected_signal = crop_center(
        expected_signal,
        (
            int(0.333 * expected_signal.shape[0]),
            int(0.333 * expected_signal.shape[1]),
        ),
    )

    # -------------------------------------------------------------------------
    # Compute plot limits (we basically only want to plot the ROI)
    # -------------------------------------------------------------------------

    with instrument_unit_context:
        outer_radius = Quantity(*config['roi_mask']['outer_radius'])
        x_diff = frame_size[0] - (int(outer_radius.to('pixel').value) * 2 + 7)
        y_diff = frame_size[1] - (int(outer_radius.to('pixel').value) * 2 + 7)
        x_lim = (int(x_diff / 2), frame_size[0] - int(x_diff / 2) - 1)
        y_lim = (int(y_diff / 2), frame_size[0] - int(y_diff / 2) - 1)

    # -------------------------------------------------------------------------
    # Finally, create the plots
    # -------------------------------------------------------------------------

    # Plot hypothesis map
    plot_hypothesis_map(
        hypotheses=hypotheses,
        contours=contours,
        plots_dir=plots_dir,
        pixscale=pixscale,
        x_lim=x_lim,
        y_lim=y_lim,
    )

    # Plot match fraction
    plot_match_fraction_map(
        match_fraction=match_fraction,
        contours=contours,
        plots_dir=plots_dir,
        pixscale=pixscale,
        x_lim=x_lim,
        y_lim=y_lim,
    )

    # Plot selection mask
    plot_selection_mask(
        selection_mask=selection_mask,
        contours=contours,
        plots_dir=plots_dir,
        pixscale=pixscale,
        x_lim=x_lim,
        y_lim=y_lim,
    )

    # Plot polar match fraction
    plot_polar_match_fraction(
        polar_match_fraction=polar_match_fraction,
        expected_signal=expected_signal,
        plots_dir=plots_dir,
    )

    # Plot cross-correlation result
    plot_template_matching(
        matched=matched,
        plots_dir=plots_dir,
        dec_rac_center=dec_rac_center,
    )

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
