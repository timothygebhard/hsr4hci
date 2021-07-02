"""
Inject fake planet and run a HSR-based post-processing pipeline.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import argparse
import os
import time

from astropy.units import Quantity

import numpy as np

from hsr4hci.base_models import BaseModelCreator
from hsr4hci.config import load_config
from hsr4hci.data import load_dataset
from hsr4hci.derotating import derotate_combine
from hsr4hci.fits import save_fits
from hsr4hci.forward_modeling import add_fake_planet
from hsr4hci.hypotheses import get_all_hypotheses
from hsr4hci.masking import get_roi_mask, get_positions_from_mask
from hsr4hci.match_fraction import get_all_match_fractions
from hsr4hci.positions import get_injection_position
from hsr4hci.psf import get_psf_fwhm
from hsr4hci.residuals import get_residual_selection_mask
from hsr4hci.training import train_all_models
from hsr4hci.units import InstrumentUnitsContext


# -----------------------------------------------------------------------------
# MAIN CODE
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    script_start = time.time()
    print('\nTRAIN HALF-SIBLING REGRESSION MODELS\n', flush=True)

    # -------------------------------------------------------------------------
    # Parse command line arguments
    # -------------------------------------------------------------------------

    # Set up parser for command line arguments
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
    # Load experiment configuration and data set
    # -------------------------------------------------------------------------

    # Get experiment directory
    experiment_dir = Path(os.path.expanduser(args.experiment_dir))
    if not experiment_dir.exists():
        raise NotADirectoryError(f'{experiment_dir} does not exist!')

    # Get path to results directory
    results_dir = experiment_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    # Load experiment config from JSON
    print('Loading experiment configuration...', end=' ', flush=True)
    config = load_config(experiment_dir / 'config.json')
    print('Done!', flush=True)

    # Load frames, parallactic angles, etc. from HDF file
    print('Loading data set...', end=' ', flush=True)
    stack, parang, psf_template, observing_conditions, metadata = load_dataset(
        **config['dataset']
    )
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Define various useful shortcuts; activate unit conversions
    # -------------------------------------------------------------------------

    # Quantities related to the size of the data set
    n_frames, x_size, y_size = stack.shape
    frame_size = (x_size, y_size)

    # Metadata of the data set
    pixscale = float(metadata['PIXSCALE'])
    lambda_over_d = float(metadata['LAMBDA_OVER_D'])

    # Other shortcuts
    selected_keys = config['observing_conditions']['selected_keys']
    n_signal_times = config['n_signal_times']

    # Define the unit conversion context for this data set
    instrument_units_context = InstrumentUnitsContext(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
    )

    # Fit the FWHM of the PSF (in pixels)
    psf_fwhm = get_psf_fwhm(psf_template)

    # -------------------------------------------------------------------------
    # Inject a fake planet into the stack
    # -------------------------------------------------------------------------

    # Get injection parameters
    contrast = config['injection']['contrast']
    separation = config['injection']['separation']
    azimuthal_position = config['injection']['azimuthal_position']

    # If any parameter is None, skip the injection...
    if contrast is None or separation is None or azimuthal_position is None:
        print('Skipping injection of a fake planet!', flush=True)

    # ... otherwise, add a fake planet with given parameters to the stack
    else:

        # Convert separation from units of FWHM to pixel
        separation *= psf_fwhm

        # Compute position at which to inject the fake planet
        print('Computing injection position...', end=' ', flush=True)
        injection_position = get_injection_position(
            separation=Quantity(separation, 'pixel'),
            azimuthal_position=azimuthal_position,
        )
        print(
            f'Done! (separation = {separation:.1f} pixel, '
            f'azimuthal_position = {azimuthal_position})',
            flush=True,
        )

        # Inject the fake planet at the injection_position
        print('Injecting fake planet...', end=' ', flush=True)
        stack = np.asarray(
            add_fake_planet(
                stack=stack,
                parang=parang,
                psf_template=psf_template,
                polar_position=injection_position,
                magnitude=contrast,
                extra_scaling=1,
                dit_stack=float(metadata['DIT_STACK']),
                dit_psf_template=float(metadata['DIT_PSF_TEMPLATE']),
                return_planet_positions=False,
                interpolation='bilinear',
            )
        )
        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # STEP 1: Train HSR models
    # -------------------------------------------------------------------------

    # Set up a BaseModelCreator to create instances of our base model
    base_model_creator = BaseModelCreator(**config['base_model'])

    # Construct the mask for the region of interest (ROI)
    with instrument_units_context:
        roi_mask = get_roi_mask(
            mask_size=frame_size,
            inner_radius=Quantity(*config['roi_mask']['inner_radius']),
            outer_radius=Quantity(*config['roi_mask']['outer_radius']),
        )

    print('\nTraining models:', flush=True)
    with instrument_units_context:
        results = train_all_models(
            roi_mask=roi_mask,
            stack=stack,
            parang=parang,
            psf_template=psf_template,
            obscon_array=observing_conditions.as_array(selected_keys),
            selection_mask_config=config['selection_mask'],
            base_model_creator=base_model_creator,
            n_train_splits=config['n_train_splits'],
            train_mode=config['train_mode'],
            n_signal_times=n_signal_times,
            n_roi_splits=1,
            roi_split=0,
            return_format='full',
        )
    residuals = dict(results['residuals'])
    print()

    # NOTE: We *do not* save the full results here!

    # -------------------------------------------------------------------------
    # STEP 2: Find hypotheses
    # -------------------------------------------------------------------------

    # Find best hypothesis for every pixel
    print('Finding best hypothesis for each spatial pixel:', flush=True)
    hypotheses, similarities = get_all_hypotheses(
        roi_mask=roi_mask,
        residuals=residuals,
        parang=parang,
        n_signal_times=n_signal_times,
        frame_size=frame_size,
        psf_template=psf_template,
    )

    # Save hypotheses as a FITS file
    print('Saving hypotheses to FITS...', end=' ', flush=True)
    file_path = results_dir / 'hypotheses.fits'
    save_fits(array=hypotheses, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # STEP 3: Compute match fractions
    # -------------------------------------------------------------------------

    # Compute match fraction for every pixel
    print('\nComputing match fraction:', flush=True)
    mean_mf, median_mf, _ = get_all_match_fractions(
        residuals=residuals,
        hypotheses=hypotheses,
        parang=parang,
        psf_template=psf_template,
        roi_mask=roi_mask,
        frame_size=frame_size,
    )

    # Save match fraction( as FITS file
    print('Saving match fraction to FITS...', end=' ', flush=True)
    file_path = results_dir / 'median_mf.fits'
    save_fits(array=median_mf, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # STEP 4: Find selection mask
    # -------------------------------------------------------------------------

    # Compute the selection mask that determines which residual type (default
    # or based on signal fitting / masking) is used for a pixel
    print('\nComputing selection mask for residuals...', end=' ', flush=True)
    selection_mask, _, _, _, _ = get_residual_selection_mask(
        match_fraction=median_mf,
        parang=parang,
        psf_template=psf_template,
    )
    print('Done!', flush=True)

    # (Always) save the selection mask
    print('Saving selection_mask mask to FITS...', end=' ', flush=True)
    array = np.array(selection_mask).astype(int)
    file_path = results_dir / 'selection_mask.fits'
    save_fits(array=array, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # STEP 5: Assemble residuals
    # -------------------------------------------------------------------------

    # Keep track of the default and the hypothesis-based residuals, as we
    # will save those to ensure that a change in the way the selection mask
    # is determined does not require re-running all experiments.
    default_residuals = np.array(residuals['default'])
    hypothesis_residuals = np.full_like(default_residuals, np.nan)

    # Loop over all pixels in the ROI and select residuals
    for (x, y) in get_positions_from_mask(roi_mask):

        # Get the hypothesis for this pixel, and store the hypothesis-based
        # residual for this pixel (i.e., the "best" residual that was obtained
        # with signal fitting / masking).
        if not np.isnan(hypotheses[x, y]):
            signal_time = str(int(hypotheses[x, y]))
            hypothesis_residuals[:, x, y] = residuals[signal_time][:, x, y]

    # Save the default residuals to FITS
    print('Saving default residuals to FITS...', end=' ', flush=True)
    file_path = results_dir / 'default_residuals.fits'
    save_fits(array=default_residuals, file_path=file_path)
    print('Done!', flush=True)

    # Save the hypothesis residuals to FITS
    print('Saving hypothesis residuals to FITS...', end=' ', flush=True)
    file_path = results_dir / 'hypothesis_residuals.fits'
    save_fits(array=hypothesis_residuals, file_path=file_path)
    print('Done!', flush=True)

    # Determine the "final" residuals, that is, the combination of the default
    # and the hypothesis-based residuals determined by the selection mask
    final_residuals = np.copy(default_residuals)
    final_residuals[:, selection_mask] = hypothesis_residuals[
        :, selection_mask
    ]

    # -------------------------------------------------------------------------
    # STEP 6: Compute signal estimate
    # -------------------------------------------------------------------------

    # Compute final signal estimate
    print('Computing signal estimate...', end=' ', flush=True)
    signal_estimate = derotate_combine(
        stack=final_residuals, parang=parang, mask=~roi_mask
    )
    print('Done!', flush=True)

    # Save the final signal estimate
    print('Saving signal estimate to FITS...', end=' ', flush=True)
    file_path = results_dir / 'signal_estimate.fits'
    save_fits(array=signal_estimate, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
