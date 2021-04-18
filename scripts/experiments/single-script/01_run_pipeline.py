"""
Run the HSR pipeline in a single script.
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
from hsr4hci.consistency_checks import get_all_match_fractions
from hsr4hci.data import load_dataset
from hsr4hci.derotating import derotate_combine
from hsr4hci.fits import save_fits
from hsr4hci.hdf import save_dict_to_hdf
from hsr4hci.hypotheses import get_all_hypotheses
from hsr4hci.masking import get_roi_mask, get_positions_from_mask
from hsr4hci.signal_estimates import get_selection_mask
from hsr4hci.training import train_all_models
from hsr4hci.units import set_units_for_instrument


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
    parser.add_argument(
        '--dont-save-intermediate-results',
        action='store_true',
        help='If this flag is set, only the final signal estimate is saved.',
    )
    args = parser.parse_args()

    # Define shortcut
    save_intermediate = not args.dont_save_intermediate_results

    # -------------------------------------------------------------------------
    # Load experiment configuration and data set
    # -------------------------------------------------------------------------

    # Get experiment directory
    experiment_dir = Path(os.path.expanduser(args.experiment_dir))
    if not experiment_dir.exists():
        raise NotADirectoryError(f'{experiment_dir} does not exist!')

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

    # Activate the unit conversions for this instrument
    set_units_for_instrument(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
        verbose=False,
    )

    # -------------------------------------------------------------------------
    # STEP 1: Train HSR models
    # -------------------------------------------------------------------------

    # Set up a BaseModelCreator to create instances of our base model
    base_model_creator = BaseModelCreator(**config['base_model'])

    # Construct the mask for the region of interest (ROI)
    roi_mask = get_roi_mask(
        mask_size=frame_size,
        inner_radius=Quantity(*config['roi_mask']['inner_radius']),
        outer_radius=Quantity(*config['roi_mask']['outer_radius']),
    )

    print('\nTraining models:', flush=True)
    results = train_all_models(
        roi_mask=roi_mask,
        stack=stack,
        parang=parang,
        psf_template=psf_template,
        obscon_array=observing_conditions.as_array(selected_keys),
        selection_mask_config=config['selection_mask'],
        base_model_creator=base_model_creator,
        n_splits=config['n_splits'],
        mode=config['mode'],
        n_signal_times=n_signal_times,
        n_roi_splits=1,
        roi_split=0,
        return_format='full',
    )
    print()

    # Save results to HDF if desired
    if save_intermediate:

        # Create a directory for the HDF files. This is slightly complicated
        # because, due to storage limitations, it should not be created in the
        # /home directory, but rather on /work, with a symlink connecting it
        # to the rest of the experiment directory.

        # First, recreate the structure of the experiment directory in /work
        work_dir = Path(experiment_dir.as_posix().replace('/home/', '/work/'))
        work_dir.mkdir(exist_ok=True, parents=True)

        # Now, create a HDF directory on /work
        work_hdf_dir = work_dir / 'hdf'
        work_hdf_dir.mkdir(exist_ok=True)
    
        # Then, create a symlink from /home to /work
        home_hdf_dir = experiment_dir / 'hdf'
        if not home_hdf_dir.exists():
            home_hdf_dir.symlink_to(work_hdf_dir, target_is_directory=True)

        print('Saving results to HDF...', end=' ', flush=True)
        file_path = home_hdf_dir / 'results.hdf'
        save_dict_to_hdf(dictionary=results, file_path=file_path)
        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # STEP 2: Find hypotheses
    # -------------------------------------------------------------------------

    # Find best hypothesis for every pixel
    print('\nFinding best hypothesis for each spatial pixel:', flush=True)
    hypotheses, similarities = get_all_hypotheses(
        roi_mask=roi_mask,
        dict_or_path=results,
        parang=parang,
        n_signal_times=n_signal_times,
        frame_size=frame_size,
        psf_template=psf_template,
    )

    # Save hypotheses and similarities, if desired
    if save_intermediate:

        # Create directory for hypothesis
        hypotheses_dir = experiment_dir / 'hypotheses'
        hypotheses_dir.mkdir(exist_ok=True)

        # Save hypotheses as a FITS file
        print('\nSaving hypotheses to FITS...', end=' ', flush=True)
        file_path = hypotheses_dir / 'hypotheses.fits'
        save_fits(array=hypotheses, file_path=file_path)
        print('Done!', flush=True)

        # Save cosine similarities (of hypotheses) as a FITS file
        print('Saving similarities to FITS...', end=' ', flush=True)
        file_path = hypotheses_dir / 'similarities.fits'
        save_fits(array=similarities, file_path=file_path)
        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # STEP 3: Compute match fractions
    # -------------------------------------------------------------------------

    # Compute match fraction for every pixel
    print('\nComputing match fraction:', flush=True)
    mean_mf, median_mf, _ = get_all_match_fractions(
        dict_or_path=results,
        hypotheses=hypotheses,
        parang=parang,
        psf_template=psf_template,
        roi_mask=roi_mask,
        frame_size=frame_size,
    )

    # Save match fractions, if desired
    if save_intermediate:

        # Create matches directory
        mf_dir = experiment_dir / 'match_fractions'
        mf_dir.mkdir(exist_ok=True)

        # Save match fraction(s) as FITS file
        print('Saving match fraction to FITS...', end=' ', flush=True)
        file_path = mf_dir / 'mean_mf.fits'
        save_fits(array=mean_mf, file_path=file_path)
        file_path = mf_dir / 'median_mf.fits'
        save_fits(array=median_mf, file_path=file_path)
        print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # STEP 4: Find selection mask
    # -------------------------------------------------------------------------

    # Compute the selection mask that determines which residual type (default
    # or based on signal fitting / masking) is used for a pixel
    print('\nComputing selection mask for residuals...', end=' ', flush=True)
    selection_mask, threshold = get_selection_mask(
        match_fraction=median_mf,
        roi_mask=roi_mask,
        filter_size=int(config['consistency_checks']['filter_size']),
    )
    print(f'Done! (threshold = {threshold:.3f})', flush=True)

    # Create results directory
    results_dir = experiment_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    # (Always) save the selection mask
    print('Saving selection_mask mask to FITS...', end=' ', flush=True)
    array = np.array(selection_mask).astype(int)
    file_path = results_dir / 'selection_mask.fits'
    save_fits(array=array, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # STEP 5: Assemble residuals
    # -------------------------------------------------------------------------

    # Initialize everything to the default residuals
    residuals = np.array(results['default']['residuals'])

    # For the pixels where the selection mask is 1, select the residuals
    # based on the corresponding hypothesis
    for (x, y) in get_positions_from_mask(selection_mask):
        signal_time = str(int(hypotheses[x, y]))
        residuals[:, x, y] = np.array(
            results[signal_time]['residuals'][:, x, y]
        )

    # -------------------------------------------------------------------------
    # STEP 6: Compute signal estimate
    # -------------------------------------------------------------------------

    # Compute final signal estimate
    print('Computing signal estimate...', end=' ', flush=True)
    signal_estimate = derotate_combine(
        stack=residuals,
        parang=parang,
        mask=~roi_mask,
    )
    print('Done!', flush=True)

    # (Always) save the final signal estimate
    print('Saving signal estimate to FITS...', end=' ', flush=True)
    file_path = results_dir / 'signal_estimate.fits'
    save_fits(array=signal_estimate, file_path=file_path)
    print('Done!', flush=True)

    # -------------------------------------------------------------------------
    # Postliminaries
    # -------------------------------------------------------------------------

    print(f'\nThis took {time.time() - script_start:.1f} seconds!\n')
