"""
Integration tests for the full HSR / PCA pipeline.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict

import json

from _pytest.tmpdir import TempPathFactory
from astropy.modeling import models
from astropy.units import Quantity

import numpy as np
import pytest

from hsr4hci.base_models import BaseModelCreator
from hsr4hci.config import load_config
from hsr4hci.data import load_dataset
from hsr4hci.derotating import derotate_combine
from hsr4hci.fits import save_fits
from hsr4hci.forward_modeling import add_fake_planet
from hsr4hci.hypotheses import get_all_hypotheses
from hsr4hci.hdf import save_dict_to_hdf
from hsr4hci.masking import get_roi_mask
from hsr4hci.metrics import compute_metrics
from hsr4hci.match_fraction import get_all_match_fractions
from hsr4hci.plotting import plot_frame
from hsr4hci.pca import get_pca_signal_estimates
from hsr4hci.psf import get_psf_fwhm
from hsr4hci.residuals import (
    assemble_residual_stack_from_hypotheses,
    get_residual_selection_mask,
)
from hsr4hci.training import train_all_models
from hsr4hci.units import InstrumentUnitsContext


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_dir(tmp_path_factory: TempPathFactory) -> Path:
    """
    Create a directory in which all the integration test data is stored.
    """

    return Path(tmp_path_factory.mktemp('integration_test', numbered=False))


@pytest.fixture(scope="session")
def data_dir(test_dir: Path) -> Path:
    """
    Create a directory for storing the mock data set.
    """

    _data_dir = test_dir / 'data'
    _data_dir.mkdir(exist_ok=True)
    return _data_dir


@pytest.fixture(scope="session")
def pca_experiment_dir(test_dir: Path) -> Path:
    """
    Create an experiment directory for the PCA pipeline.
    """

    _experiment_dir = test_dir / 'pca_experiment_dir'
    _experiment_dir.mkdir(exist_ok=True)
    return _experiment_dir


@pytest.fixture(scope="session")
def hsr_experiment_dir(test_dir: Path) -> Path:
    """
    Create an experiment directory for the HSR pipeline.
    """

    _experiment_dir = test_dir / 'hsr_experiment_dir'
    _experiment_dir.mkdir(exist_ok=True)
    return _experiment_dir


@pytest.fixture(scope="session")
def test_data_path(data_dir: Path) -> Path:
    """
    Create an HDF file with a mock data set.
    """

    np.random.seed(42)

    # Define global parameters
    n_frames, x_size, y_size = (50, 25, 25)

    # Create fake PSF template
    x, y = np.meshgrid(np.arange(33), np.arange(33))
    gaussian = models.Gaussian2D(x_mean=16, y_mean=16)
    psf_template = gaussian(x, y)

    # Create fake parallactic angles
    parang = np.linspace(17, 137, n_frames)

    # Create a fake stack
    stack = np.random.normal(0, 1, (n_frames, x_size, y_size))
    stack += np.rot90(stack, k=2, axes=(1, 2))
    stack = add_fake_planet(
        stack=stack,
        parang=parang,
        psf_template=psf_template,
        polar_position=(Quantity(5, 'pixel'), Quantity(45, 'degree')),
        magnitude=1,
        extra_scaling=15,
        dit_stack=1,
        dit_psf_template=1,
        return_planet_positions=False,
    )

    # Create fake observing conditions
    observing_conditions = {
        'array_1': np.random.normal(0, 1, n_frames),
        'array_2': np.random.normal(0, 1, n_frames),
        'array_3': np.random.normal(0, 1, n_frames),
    }

    # Create fake meta data
    metadata = {
        "CENTRAL_LAMBDA": "3800 nm",
        "CORONAGRAPH": "",
        "DATE": "2021-05-31",
        "DIT_PSF_TEMPLATE": 1,
        "DIT_STACK": 1,
        "ESO_PROGRAM_ID": "UNKNOWN",
        "FILTER": "L'",
        "INSTRUMENT": "NACO",
        "LAMBDA_OVER_D": 0.0956,
        "ND_FILTER": 1.0,
        "PIXSCALE": 0.0271,
        "TARGET_STAR": "TEST",
    }

    # Collect everything in a properly structured dict
    test_data = {
        'stack': stack,
        'parang': parang,
        'psf_template': psf_template,
        'metadata': metadata,
        'observing_conditions': {'interpolated': observing_conditions},
    }

    # Save the test data to an HDF file
    file_path = data_dir / 'test_data.hdf'
    save_dict_to_hdf(dictionary=test_data, file_path=file_path)

    return file_path


@pytest.fixture(scope="session")
def hsr_config_file(hsr_experiment_dir: Path, test_data_path: Path) -> None:
    """
    Create a mock config file for the HSR pipeline.
    """

    config_dict = {
        "dataset": {
            "name_or_path": test_data_path.as_posix(),
            "binning_factor": 1,
            "frame_size": [23, 23],
        },
        "observing_conditions": {"selected_keys": "all"},
        "roi_mask": {
            "inner_radius": [0, "pixel"],
            "outer_radius": [10, "pixel"],
        },
        "selection_mask": {
            "radius_position": [8.0, "pixel"],
            "radius_opposite": [0.2168, "arcsec"],
            "radius_excluded": [4.0, "pixel"],
        },
        "train_mode": "signal_masking",
        "n_signal_times": 10,
        "n_train_splits": 3,
        "base_model": {
            "module": "sklearn.linear_model",
            "class": "RidgeCV",
            "parameters": {"fit_intercept": True, "alphas": [1e-1, 1e2, 16]},
        },
    }

    file_path = hsr_experiment_dir / 'config.json'
    with open(file_path, 'w') as json_file:
        json.dump(config_dict, json_file, indent=2)


@pytest.fixture(scope="session")
def pca_config_file(pca_experiment_dir: Path, test_data_path: Path) -> None:
    """
    Create a mock config file for the PCA pipeline.
    """

    config_dict = {
        "dataset": {
            "name_or_path": test_data_path.as_posix(),
            "binning_factor": 1,
            "frame_size": [23, 23],
        },
        "roi_mask": {
            "inner_radius": [0, "pixel"],
            "outer_radius": [10, "pixel"],
        },
        "pca": {"min_n": 1, "max_n": 10, "default_n": 5},
    }

    file_path = pca_experiment_dir / 'config.json'
    with open(file_path, 'w') as json_file:
        json.dump(config_dict, json_file, indent=2)


def test__integration_pca(
    pca_config_file: None,
    pca_experiment_dir: Path,
) -> None:
    """
    Integration test that runs an entire PCA pipeline on mock data.
    """

    # -------------------------------------------------------------------------
    # Load experiment configuration and data
    # -------------------------------------------------------------------------

    config = load_config(pca_experiment_dir / 'config.json')
    stack, parang, psf_template, observing_conditions, metadata = load_dataset(
        **config['dataset']
    )

    n_frames, x_size, y_size = stack.shape
    frame_size = (x_size, y_size)

    min_n = int(config['pca']['min_n'])
    max_n = int(config['pca']['max_n'])
    n_components = [int(_) for _ in np.arange(min_n, max_n)]
    default_n = int(config['pca']['default_n'])
    default_idx = default_n - min_n

    pixscale = float(metadata['PIXSCALE'])
    lambda_over_d = float(metadata['LAMBDA_OVER_D'])

    instrument_unit_context = InstrumentUnitsContext(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
    )

    with instrument_unit_context:
        roi_mask = get_roi_mask(
            mask_size=frame_size,
            inner_radius=Quantity(*config['roi_mask']['inner_radius']),
            outer_radius=Quantity(*config['roi_mask']['outer_radius']),
        )

    # -------------------------------------------------------------------------
    # STEP 1: Run PCA to get signal estimates and principal components
    # -------------------------------------------------------------------------

    # Run PCA to get signal estimates and principal components
    signal_estimates, principal_components = get_pca_signal_estimates(
        stack=stack,
        parang=parang,
        n_components=n_components,
        return_components=True,
        roi_mask=None,
    )
    assert np.isclose(np.nansum(signal_estimates), 4.5674645930457345)
    assert np.isclose(np.nansum(principal_components), -4.289586, atol=1e-3)

    # Apply ROI mask after PCA
    signal_estimates[:, ~roi_mask] = np.nan

    # Select "default" signal estimate
    signal_estimate = signal_estimates[default_idx]

    # -------------------------------------------------------------------------
    # STEP 2: Compute metrics
    # -------------------------------------------------------------------------

    psf_fwhm = get_psf_fwhm(psf_template)
    assert np.isclose(psf_fwhm, 2 * np.sqrt(2 * np.log(2)))

    with instrument_unit_context:
        metrics, positions = compute_metrics(
            frame=signal_estimate,
            polar_position=(
                Quantity(0.1355, 'arcsecond'),
                Quantity(45, 'degree'),
            ),
            aperture_radius=Quantity(psf_fwhm / 2, 'pixel'),
        )
    assert np.isclose(metrics['snr']['min'], 6.593299696334281)
    assert np.isclose(metrics['snr']['max'], 14.093937863225246)
    assert np.isclose(metrics['snr']['mean'], 9.62031939013815)

    # -------------------------------------------------------------------------
    # STEP 3: Create a plot
    # -------------------------------------------------------------------------

    file_path = pca_experiment_dir / 'signal_estimate.pdf'
    plot_frame(
        frame=signal_estimate,
        file_path=file_path,
        aperture_radius=psf_fwhm,
        pixscale=float(metadata['PIXSCALE']),
        positions=[positions['final']['cartesian']],
        labels=[f'SNR={metrics["snr"]["mean"]:.1f}'],
        add_colorbar=True,
        use_logscale=False,
    )


def test__integration_hsr(
    hsr_config_file: None,
    hsr_experiment_dir: Path,
) -> None:
    """
    Integration test that runs an entire HSR pipeline on mock data.
    """

    # -------------------------------------------------------------------------
    # STEP 0: Load data, define shortcuts, set up unit conversions
    # -------------------------------------------------------------------------

    config = load_config(hsr_experiment_dir / 'config.json')
    stack, parang, psf_template, observing_conditions, metadata = load_dataset(
        **config['dataset']
    )

    _, x_size, y_size = stack.shape
    frame_size = (x_size, y_size)

    pixscale = float(metadata['PIXSCALE'])
    lambda_over_d = float(metadata['LAMBDA_OVER_D'])

    selected_keys = config['observing_conditions']['selected_keys']
    n_signal_times = config['n_signal_times']

    instrument_unit_context = InstrumentUnitsContext(
        pixscale=Quantity(pixscale, 'arcsec / pixel'),
        lambda_over_d=Quantity(lambda_over_d, 'arcsec'),
    )

    with instrument_unit_context:
        roi_mask = get_roi_mask(
            mask_size=frame_size,
            inner_radius=Quantity(*config['roi_mask']['inner_radius']),
            outer_radius=Quantity(*config['roi_mask']['outer_radius']),
        )

    # -------------------------------------------------------------------------
    # STEP 1: Train HSR models
    # -------------------------------------------------------------------------

    with instrument_unit_context:
        results = train_all_models(
            roi_mask=roi_mask,
            stack=stack,
            parang=parang,
            psf_template=psf_template,
            obscon_array=observing_conditions.as_array(selected_keys),
            selection_mask_config=config['selection_mask'],
            base_model_creator=BaseModelCreator(**config['base_model']),
            max_oc_correlation=1.0,
            n_train_splits=config['n_train_splits'],
            train_mode=config['train_mode'],
            n_signal_times=n_signal_times,
            n_roi_splits=1,
            roi_split=0,
            return_format='full',
        )
    residuals: Dict[str, np.ndarray] = dict(results['residuals'])

    print("\n\n\n")
    print("np.nansum(residuals['default']) =", np.nansum(residuals['default']))
    assert np.isclose(np.nansum(residuals['default']), 97.77813)
    print("np.nansum(residuals['0']) =", np.nansum(residuals['0']))
    assert np.isclose(np.nansum(residuals['0']), 257.0215)

    # -------------------------------------------------------------------------
    # STEP 2: Find hypotheses
    # -------------------------------------------------------------------------

    hypotheses, similarities = get_all_hypotheses(
        roi_mask=roi_mask,
        residuals=residuals,
        parang=parang,
        n_signal_times=n_signal_times,
        frame_size=frame_size,
        psf_template=psf_template,
    )
    print("np.nansum(hypotheses) =", np.nansum(hypotheses))
    assert np.nansum(hypotheses) == 7170.0
    print("np.nansum(similarities) =", np.nansum(similarities))
    assert np.isclose(np.nansum(similarities), 88.142)

    file_path = hsr_experiment_dir / 'hypotheses.fits'
    save_fits(hypotheses, file_path)

    # -------------------------------------------------------------------------
    # STEP 3: Compute match fractions
    # -------------------------------------------------------------------------

    mean_mf, median_mf, _ = get_all_match_fractions(
        residuals=residuals,
        hypotheses=hypotheses,
        parang=parang,
        psf_template=psf_template,
        roi_mask=roi_mask,
        frame_size=frame_size,
    )

    print("np.nansum(mean_mf) =", np.nansum(mean_mf))
    assert np.isclose(np.nansum(mean_mf), 22.525827083200713)
    print("np.nansum(median_mf) =", np.nansum(median_mf))
    assert np.isclose(np.nansum(median_mf), 23.216906441932153)

    file_path = hsr_experiment_dir / 'mean_mf.fits'
    save_fits(mean_mf, file_path)

    # -------------------------------------------------------------------------
    # STEP 4: Find selection mask
    # -------------------------------------------------------------------------

    selection_mask, _, _, _, _ = get_residual_selection_mask(
        match_fraction=mean_mf,
        parang=parang,
        psf_template=psf_template,
    )
    print("np.nansum(selection_mask) =", np.nansum(selection_mask))
    assert np.nansum(selection_mask) == 51

    file_path = hsr_experiment_dir / 'selection_mask.fits'
    save_fits(selection_mask, file_path)

    # -------------------------------------------------------------------------
    # STEP 5: Assemble residual stack and compute signal estimate
    # -------------------------------------------------------------------------

    residual_stack = assemble_residual_stack_from_hypotheses(
        residuals=residuals,
        hypotheses=hypotheses,
        selection_mask=selection_mask,
    )
    print("np.nansum(residual_stack) =", np.nansum(residual_stack))
    assert np.isclose(np.nansum(residual_stack), 1646.7012)

    file_path = hsr_experiment_dir / 'residual_stack.fits'
    save_fits(residual_stack, file_path)

    signal_estimate = derotate_combine(
        stack=residual_stack, parang=parang, mask=~roi_mask
    )
    print("np.nansum(signal_estimate) =", np.nansum(signal_estimate))
    assert np.isclose(np.nansum(signal_estimate), 32.94099403114524)

    file_path = hsr_experiment_dir / 'signal_estimate.fits'
    save_fits(signal_estimate, file_path)

    # -------------------------------------------------------------------------
    # STEP 6: Compute metrics
    # -------------------------------------------------------------------------

    psf_fwhm = get_psf_fwhm(psf_template)
    assert np.isclose(psf_fwhm, 2 * np.sqrt(2 * np.log(2)))

    with instrument_unit_context:
        metrics, positions = compute_metrics(
            frame=signal_estimate,
            polar_position=(
                Quantity(0.1355, 'arcsecond'),
                Quantity(45, 'degree'),
            ),
            aperture_radius=Quantity(psf_fwhm / 2, 'pixel'),
        )

    print("metrics['snr']['min'] =", metrics['snr']['min'])
    assert np.isclose(metrics['snr']['min'], 27.430971199560332)
    print("metrics['snr']['max'] =", metrics['snr']['max'])
    assert np.isclose(metrics['snr']['max'], 51.611312579520906)
    print("metrics['snr']['mean'] =", metrics['snr']['mean'])
    assert np.isclose(metrics['snr']['mean'], 35.993992202839024)

    # -------------------------------------------------------------------------
    # STEP 7: Create a plot
    # -------------------------------------------------------------------------

    file_path = hsr_experiment_dir / 'signal_estimate.pdf'
    plot_frame(
        frame=signal_estimate,
        file_path=file_path,
        aperture_radius=psf_fwhm,
        pixscale=float(metadata['PIXSCALE']),
        positions=[positions['final']['cartesian']],
        labels=[f'SNR={metrics["snr"]["mean"]:.1f}'],
        add_colorbar=True,
        use_logscale=False,
    )
    print('Result directory:', hsr_experiment_dir)
