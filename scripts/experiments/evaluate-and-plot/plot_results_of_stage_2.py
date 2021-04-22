"""
Plot the results of stage 2: hypothesis map, match fraction map, and the
selection mask for the residuals.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import os
import time

from astropy.units import Quantity
from skimage.measure import find_contours

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
from hsr4hci.masking import get_roi_mask
from hsr4hci.plotting import disable_ticks, get_cmap
from hsr4hci.signal_estimates import get_selection_mask
from hsr4hci.units import set_units_for_instrument


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

    # Define quantities related to the size of the data set
    n_frames = len(parang)
    frame_size = (
        int(config['dataset']['frame_size'][0]),
        int(config['dataset']['frame_size'][1]),
    )
    center = get_center(frame_size)

    # Activate the unit conversions for this instrument
    set_units_for_instrument(
        pixscale=Quantity(metadata['PIXSCALE'], 'arcsec / pixel'),
        lambda_over_d=Quantity(metadata['LAMBDA_OVER_D'], 'arcsec'),
        verbose=False,
    )

    # Construct the mask for the region of interest (ROI)
    roi_mask = get_roi_mask(
        mask_size=frame_size,
        inner_radius=Quantity(*config['roi_mask']['inner_radius']),
        outer_radius=Quantity(*config['roi_mask']['outer_radius']),
    )

    # -------------------------------------------------------------------------
    # Compute a mask for the pixels that we know should contain planet signal
    # -------------------------------------------------------------------------

    print('Determining affected pixels...', end=' ', flush=True)

    # Make sure the PSF template is correctly normalized
    psf_template -= np.min(psf_template)
    psf_template /= np.max(psf_template)

    # Down-sample the parallactic angle (to speed up the computation)
    n = n_frames // 100
    parang_resampled = parang[::n]

    # Initialize the hypothesized stack and mask of affected pixels
    hypothesized_stack = np.zeros((len(parang_resampled),) + frame_size)

    # Loop over (potentially multiple) planets and compute their stack
    for name, parameters in planets.items():
        signal_stack = np.array(
            add_fake_planet(
                stack=hypothesized_stack,
                parang=parang_resampled,
                psf_template=psf_template,
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
    affected_mask = np.max(hypothesized_stack, axis=0) # > 0.2

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Load the hypotheses map and plot it
    # -------------------------------------------------------------------------

    # Load the hypotheses from FITS
    print('\nLoading hypothesis map...', end=' ', flush=True)
    file_path = experiment_dir / 'hypotheses' / 'hypotheses.fits'
    hypotheses = np.asarray(read_fits(file_path))
    hypotheses[~roi_mask] = np.nan
    print('Done!', flush=True)

    # Create a plot of the hypothesis map
    print('Plotting hypothesis map...', end=' ', flush=True)

    # Prepare grid for the pcolormesh()
    x_range = np.arange(hypotheses.shape[0])
    y_range = np.arange(hypotheses.shape[1])
    x, y = np.meshgrid(x_range, y_range)

    # Prepare the plot
    fig, ax = plt.subplots(figsize=(4, 4))
    disable_ticks(ax)

    # Plot the hypotheses
    img = ax.pcolormesh(
        x,
        y,
        hypotheses,
        shading='nearest',
        cmap=get_cmap('viridis'),
        rasterized=True,
    )
    ax.plot(center[0], center[1], '+', color='red')

    # Overlay the region of affected pixels
    contours = find_contours(affected_mask, 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color='red', lw=2)

    # Save the results
    fig.tight_layout()
    file_path = plots_dir / 'hypotheses.pdf'
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Load the match fraction map and plot it
    # -------------------------------------------------------------------------

    # Load the match fraction from FITS
    print('\nLoading match fraction map...', end=' ', flush=True)
    file_path = experiment_dir / 'match_fractions' / 'median_mf.fits'
    median_mf = np.asarray(read_fits(file_path))
    median_mf[~roi_mask] = np.nan
    print('Done!', flush=True)

    # Create a plot of the match fraction map
    print('Plotting match fraction map...', end=' ', flush=True)

    # Prepare grid for the pcolormesh()
    x_range = np.arange(median_mf.shape[0])
    y_range = np.arange(median_mf.shape[1])
    x, y = np.meshgrid(x_range, y_range)

    # Prepare the plot
    fig, ax = plt.subplots(figsize=(4, 4))
    disable_ticks(ax)

    # Plot the match fraction
    img = ax.pcolormesh(
        x,
        y,
        median_mf,
        shading='nearest',
        cmap=get_cmap('viridis'),
        rasterized=True,
    )
    ax.plot(center[0], center[1], '+', color='red')

    # Overlay the region of affected pixels
    contours = find_contours(affected_mask, 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color='red', lw=2)

    # Save the results
    fig.tight_layout()
    file_path = plots_dir / 'match_fraction.pdf'
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Get the selection mask and plot it
    # -------------------------------------------------------------------------

    # Compute the selection mask
    print('\nComputing selection mask...', end=' ', flush=True)
    selection_mask, _ = get_selection_mask(median_mf, roi_mask=roi_mask)
    selection_mask = selection_mask.astype(float)
    selection_mask[~roi_mask] = np.nan
    print('Done!', flush=True)

    # Create a plot of the selection mask
    print('Plotting selection mask...', end=' ', flush=True)

    # Prepare grid for the pcolormesh()
    x_range = np.arange(selection_mask.shape[0])
    y_range = np.arange(selection_mask.shape[1])
    x, y = np.meshgrid(x_range, y_range)

    # Prepare the plot
    fig, ax = plt.subplots(figsize=(4, 4))
    disable_ticks(ax)

    # Plot the match fraction
    img = ax.pcolormesh(
        x,
        y,
        selection_mask,
        shading='nearest',
        cmap=get_cmap('viridis'),
        rasterized=True,
    )
    ax.plot(center[0], center[1], '+', color='red')

    # Overlay the region of affected pixels
    contours = find_contours(affected_mask, 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color='red', lw=2)

    # Save the results
    fig.tight_layout()
    file_path = plots_dir / 'selection_mask.pdf'
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)

    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
