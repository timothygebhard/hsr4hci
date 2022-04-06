"""
Plot all contrast curves.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hsr4hci.config import get_experiments_dir
from hsr4hci.contrast import get_contrast_curve
from hsr4hci.data import load_psf_template, load_metadata
from hsr4hci.plotting import set_fontsize, adjust_luminosity
from hsr4hci.psf import get_psf_fwhm


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nPLOT ALL CONTRAST CURVES\n')

    # -------------------------------------------------------------------------
    # Parse command line arguments
    # -------------------------------------------------------------------------

    # Set up a parser and parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Name of the data set, e.g., "beta_pictoris__lp".',
    )
    parser.add_argument(
        '--min-contrast',
        type=float,
        default=6.0,
        help='Minimum contrast / upper limit for plot.',
    )
    parser.add_argument(
        '--max-contrast',
        type=float,
        default=11.0,
        help='Maximum contrast / lower limit for plot.',
    )
    parser.add_argument(
        '--min-separation',
        type=float,
        default=2.0,
        help='Minimum separation / lower limit for plot.',
    )
    parser.add_argument(
        '--max-separation',
        type=float,
        default=8.0,
        help='Maximum separation / upper limit for plot.',
    )
    parser.add_argument(
        '--sigma-threshold',
        type=float,
        default=5.0,
        help='(Gaussian) sigma for contrast curve. Default: 5.',
    )
    parser.add_argument(
        '--mode',
        choices=['classic', 'alternative'],
        default='classic',
        help=(
            'How to compute the throughput: using the "classic" approach'
            'where the no_fake_planets residual is subtracted, or the '
            'alternative approach that estimate the background from other '
            'positions at the same separation.'
        ),
    )
    args = parser.parse_args()

    # Define shortcuts
    dataset = args.dataset
    mode = args.mode
    min_contrast = args.min_contrast
    max_contrast = args.max_contrast
    min_separation = args.min_separation
    max_separation = args.max_separation
    sigma_threshold = args.sigma_threshold

    # Get path to main directory
    main_dir = (
        get_experiments_dir()
        / 'main'
        / '5.3_photometry-artificial-planets'
        / dataset
    ).resolve()
    if not main_dir.exists():
        raise RuntimeError(f'{main_dir} does not exist!')

    # -------------------------------------------------------------------------
    # Get the PSF FWHM of the respective data set; load the pixscale
    # -------------------------------------------------------------------------

    # Load PSF and determine FWHM
    psf_template = load_psf_template(name_or_path=dataset)
    psf_fwhm = get_psf_fwhm(psf_template=psf_template)

    # Load metadata and get pixscale
    metadata = load_metadata(name_or_path=dataset)
    pixscale = float(metadata['PIXSCALE'])

    # -------------------------------------------------------------------------
    # Loop over different result files and plot contrast curves
    # -------------------------------------------------------------------------

    # Create a new figure and adjust margins
    fig, ax = plt.subplots(figsize=(5.8 / 2.54, 5.8 / 2.54))
    fig.subplots_adjust(left=0.155, right=0.995, top=0.865, bottom=0.135)

    # Loop over different algorithms
    for name, path, color, ls in [
        ('PCA (n=5)', 'pca-5', 'darkgreen', '-'),
        ('PCA (n=20)', 'pca-20', 'forestgreen', '-'),
        ('PCA (n=50)', 'pca-50', 'lawngreen', '-'),
        ('HSR (SF)', 'signal_fitting', 'C0', '-'),
        ('HSR (SM)', 'signal_masking', 'C1', '-'),
        ('HSR (SF+OC)', 'signal_fitting__oc', 'C0', '--'),
        ('HSR (SM+OC)', 'signal_masking__oc', 'C1', '--'),
    ]:

        print(f'Plotting {name}...', end=' ', flush=True)

        # Read in result
        file_path = main_dir / path / f'results__{mode}.tsv'
        if not file_path.exists():
            print('Skipped!', flush=True)
            continue
        else:
            df = pd.read_csv(file_path, sep='\t')

        # Apply separation limits on the data frame
        df = df[
            (df.separation >= min_separation)
            & (df.separation <= max_separation)
        ]

        # Compute the contrast curve
        separations, detection_limits = get_contrast_curve(
            df=df,
            sigma_threshold=sigma_threshold,
            log_transform=True,
        )

        # Define additional arguments for plotting
        if ls == '--':
            plot_args = dict(dash_capstyle='round', dashes=(1, 2), color=color)
        else:
            plot_args = dict(color=adjust_luminosity(color))

        # Plot the contrast curve
        ax.plot(separations, detection_limits, ls=ls, label=name, **plot_args)

        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Add legend to the plot and save the plot
    # -------------------------------------------------------------------------

    # Add a second x-axis on top with the separation in units of pixels
    ax2 = ax.twiny()

    # Add axes labels
    ax.set_ylabel('Contrast (in magnitudes)')
    ax.set_xlabel('Separation (in units of the PSF FWHM)')
    ax2.set_xlabel('Separation (in arcsec)')

    # Set x-limits and ticks
    x_lower, x_upper = min_separation - 0.25, max_separation + 0.25
    ax.set_xlim(x_lower, x_upper)
    ax.set_xticks(np.arange(min_separation, max_separation + 1))
    ax2.set_xlim(x_lower * psf_fwhm * pixscale, x_upper * psf_fwhm * pixscale)
    ax2.set_xticks(
        np.arange(
            round(x_lower * psf_fwhm * pixscale, 1) + 0.1,
            min(round(x_upper * psf_fwhm * pixscale, 1), 1.0) + 0.1,
            0.1,
        )
    )

    # Set y-limits and ticks
    ax.set_ylim(max_contrast + 0.5, min_contrast - 0.5)
    ax.set_yticks(np.arange(min_contrast, max_contrast + 1))

    # Adjust fontsize
    set_fontsize(ax=ax, fontsize=6)
    set_fontsize(ax=ax2, fontsize=6)

    # Add legend
    ax.legend(
        loc='upper right',
        fontsize=6,
        frameon=True,
        handlelength=0.8,
    )

    # Save plot as PDF
    file_name = f'all-contrast-curves__{sigma_threshold:.1f}-sigma__{mode}.pdf'
    plt.savefig(main_dir / file_name, dpi=600)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
