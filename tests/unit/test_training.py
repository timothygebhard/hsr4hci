"""
Tests for training.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any

import numpy as np
import pytest

# noinspection PyProtectedMember
from hsr4hci.training import (
    _train_default_model,
    _train_signal_fitting_model,
    _train_signal_masking_model,
    get_signal_times,
    train_all_models,
    train_model_for_position,
)
from hsr4hci.base_models import BaseModelCreator
from hsr4hci.masking import get_annulus_mask


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

def test__get_signal_times() -> None:

    # Case 1
    signal_times = get_signal_times(n_frames=10, n_signal_times=1)
    assert np.array_equal(signal_times, np.array([0]))

    # Case 2
    signal_times = get_signal_times(n_frames=10, n_signal_times=2)
    assert np.array_equal(signal_times, np.array([0, 9]))

    # Case 3
    signal_times = get_signal_times(n_frames=10, n_signal_times=3)
    assert np.array_equal(signal_times, np.array([0, 4, 9]))

    # Case 4
    signal_times = get_signal_times(n_frames=10, n_signal_times=0)
    assert np.array_equal(signal_times, np.array([]))


def test___train_default_model() -> None:

    base_model_config = {
        'module': 'sklearn.linear_model',
        'class': 'LinearRegression',
        'parameters': {},
    }
    base_model_creator = BaseModelCreator(**base_model_config)

    # Case 1
    np.random.seed(42)
    coefficients = np.array([1, 2])
    train_predictors = np.random.normal(0, 1, (10, 2))
    train_targets = train_predictors @ coefficients
    model: Any = _train_default_model(
        base_model_creator=base_model_creator,
        train_predictors=train_predictors,
        train_targets=train_targets,
    )
    assert model is not None
    assert np.allclose(model.coef_, np.array([[1, 2]]))
    assert np.isclose(float(model.intercept_), 0)


def test___train_signal_fitting_model() -> None:

    illegal_base_model_config = {
        'module': 'sklearn.ensemble',
        'class': 'RandomForestRegressor',
        'parameters': {},
    }
    illegal_base_model_creator = BaseModelCreator(**illegal_base_model_config)

    base_model_config = {
        'module': 'sklearn.linear_model',
        'class': 'LinearRegression',
        'parameters': {},
    }
    base_model_creator = BaseModelCreator(**base_model_config)

    # Case 1
    with pytest.raises(RuntimeError) as runtime_error:
        _train_signal_fitting_model(
            base_model_creator=illegal_base_model_creator,
            train_predictors=np.empty((5, 5)),
            train_targets=np.empty(5),
            expected_signal=np.empty(5),
        )
    assert 'Signal fitting only works with linear' in str(runtime_error)

    # Case 2
    np.random.seed(423)
    coefficients = np.array([1, 2, 3])
    expected_signal = np.zeros(101)
    expected_signal[45:55] = 5
    train_predictors = np.random.normal(0, 1, (101, 3))
    train_targets = train_predictors @ coefficients + 1.5 * expected_signal
    model: Any
    model, planet_coefficient = _train_signal_fitting_model(
        base_model_creator=base_model_creator,
        train_predictors=train_predictors,
        train_targets=train_targets,
        expected_signal=expected_signal,
    )
    assert model is not None
    assert np.allclose(model.coef_, np.array([1, 2, 3]))
    assert np.isclose(float(model.intercept_), 0, atol=1e-7)
    assert np.isclose(planet_coefficient, 1.5)

    # Case 3
    np.random.seed(423)
    coefficients = np.array([1, 2, 3])
    expected_signal = np.zeros(101)
    expected_signal[45:55] = 5
    train_predictors = np.random.normal(0, 1, (101, 3))
    train_targets = train_predictors @ coefficients + -1 * expected_signal
    model, planet_coefficient = _train_signal_fitting_model(
        base_model_creator=base_model_creator,
        train_predictors=train_predictors,
        train_targets=train_targets,
        expected_signal=expected_signal,
    )
    assert model is not None
    assert np.allclose(
        model.coef_, np.array([1.18788776, 1.79318367, 3.04629971])
    )
    assert np.isclose(float(model.intercept_), -0.496544608313041)
    assert np.isclose(planet_coefficient, -1)


def test___train_signal_masking_model() -> None:

    base_model_config = {
        'module': 'sklearn.linear_model',
        'class': 'LinearRegression',
        'parameters': {},
    }
    base_model_creator = BaseModelCreator(**base_model_config)

    # Case 1
    np.random.seed(42)
    coefficients = np.array([1, 2, 3])
    expected_signal = np.zeros(101)
    expected_signal[45:55] = 5
    train_predictors = np.random.normal(0, 1, (101, 3))
    train_targets = train_predictors @ coefficients + expected_signal
    model: Any = _train_signal_masking_model(
        base_model_creator=base_model_creator,
        train_predictors=train_predictors,
        train_targets=train_targets,
        expected_signal=expected_signal,
    )
    assert model is not None
    assert np.allclose(model.coef_, np.array([[1, 2, 3]]))
    assert np.isclose(float(model.intercept_), 0)

    # Case 2
    np.random.seed(42)
    coefficients = np.array([1, 2, 3])
    expected_signal = np.zeros(101)
    expected_signal[15:85] = 5
    train_predictors = np.random.normal(0, 1, (101, 3))
    train_targets = train_predictors @ coefficients + expected_signal
    model = _train_signal_masking_model(
        base_model_creator=base_model_creator,
        train_predictors=train_predictors,
        train_targets=train_targets,
        expected_signal=expected_signal,
    )
    assert model is None


def test__train_model_for_position() -> None:

    base_model_config = {
        'module': 'sklearn.linear_model',
        'class': 'LinearRegression',
        'parameters': {},
    }
    base_model_creator = BaseModelCreator(**base_model_config)

    # Create a stack that has perfect mirror symmetry so it can be predicted
    # perfectly by our model
    stack = np.random.normal(0, 1, (50, 31, 31))
    stack += np.rot90(stack, k=2, axes=(1, 2))
    parang = np.linspace(0, 50, 50)
    obscon_array = np.random.normal(0, 1, (50, 5))
    position = (11, 8)
    selection_mask_config = {
        'radius_position': (2, 'pixel'),
        'radius_opposite': (2, 'pixel'),
    }
    psf_template = np.array([[0, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 0]])
    n_train_splits = 1

    # Case 1
    residual, model_parameters = train_model_for_position(
        stack=stack,
        parang=parang,
        obscon_array=obscon_array,
        position=position,
        train_mode='default',
        signal_time=None,
        selection_mask_config=selection_mask_config,
        psf_template=psf_template,
        n_train_splits=n_train_splits,
        base_model_creator=base_model_creator,
    )
    assert np.isclose(np.sum(residual), 0)
    assert np.isclose(np.sum(model_parameters['pixel_coefs']), 1)

    # Case 2
    residual, model_parameters = train_model_for_position(
        stack=stack,
        parang=parang,
        obscon_array=obscon_array,
        position=position,
        train_mode='signal_fitting',
        signal_time=0,
        selection_mask_config=selection_mask_config,
        psf_template=psf_template,
        n_train_splits=n_train_splits,
        base_model_creator=base_model_creator,
    )
    assert np.isclose(np.sum(residual), 0)
    assert np.isclose(np.sum(model_parameters['pixel_coefs']), 1)
    assert np.isclose(np.sum(model_parameters['planet_coefs']), 0)

    # Case 3
    residual, model_parameters = train_model_for_position(
        stack=stack,
        parang=parang,
        obscon_array=obscon_array,
        position=position,
        train_mode='signal_masking',
        signal_time=0,
        selection_mask_config=selection_mask_config,
        psf_template=psf_template,
        n_train_splits=n_train_splits,
        base_model_creator=base_model_creator,
    )
    assert np.isclose(np.sum(residual), 0)
    assert np.isclose(np.sum(model_parameters['pixel_coefs']), 1)

    # Case 4
    with pytest.raises(RuntimeError) as runtime_error:
        train_model_for_position(
            stack=stack,
            parang=parang,
            obscon_array=obscon_array,
            position=position,
            train_mode='signal_masking',
            signal_time=None,
            selection_mask_config=selection_mask_config,
            psf_template=psf_template,
            n_train_splits=n_train_splits,
            base_model_creator=base_model_creator,
        )
    assert 'signal_time must not be None!' in str(runtime_error)

    # Case 5
    with pytest.raises(ValueError) as value_error:
        train_model_for_position(
            stack=stack,
            parang=parang,
            obscon_array=obscon_array,
            position=position,
            train_mode='illegal',
            signal_time=None,
            selection_mask_config=selection_mask_config,
            psf_template=psf_template,
            n_train_splits=n_train_splits,
            base_model_creator=base_model_creator,
        )
    assert 'Illegal value for train_mode!' in str(value_error)

    base_model_config = {
        'module': 'sklearn.linear_model',
        'class': 'RidgeCV',
        'parameters': {'alphas': [1e-8, 1e0, 9]},
    }
    base_model_creator = BaseModelCreator(**base_model_config)

    # Case 6
    residual, model_parameters = train_model_for_position(
        stack=stack,
        parang=parang,
        obscon_array=obscon_array,
        position=position,
        train_mode='signal_masking',
        signal_time=25,
        selection_mask_config=selection_mask_config,
        psf_template=psf_template,
        n_train_splits=n_train_splits,
        base_model_creator=base_model_creator,
    )
    assert np.isclose(np.sum(residual), 0, atol=1e-4)
    assert np.isclose(np.sum(model_parameters['pixel_coefs']), 1)
    assert np.isclose(np.sum(model_parameters['alphas']), 1e-8)


def test__train_all_models() -> None:

    base_model_config = {
        'module': 'sklearn.linear_model',
        'class': 'LinearRegression',
        'parameters': {},
    }
    base_model_creator = BaseModelCreator(**base_model_config)

    frame_size = (15, 15)
    stack = np.random.normal(0, 1, (50,) + frame_size)
    stack += np.rot90(stack, k=2, axes=(1, 2))
    parang = np.linspace(0, 50, 50)
    obscon_array = np.random.normal(0, 1, (50, 5))
    selection_mask_config = {
        'radius_position': (2, 'pixel'),
        'radius_opposite': (2, 'pixel'),
    }
    psf_template = np.array([[0, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 0]])
    roi_mask = get_annulus_mask(frame_size, inner_radius=2, outer_radius=6)

    # Case 1
    with pytest.raises(ValueError) as value_error:
        train_all_models(
            roi_mask=roi_mask,
            stack=stack,
            parang=parang,
            obscon_array=obscon_array,
            selection_mask_config=selection_mask_config,
            base_model_creator=base_model_creator,
            n_signal_times=3,
            psf_template=psf_template,
            n_train_splits=2,
            train_mode='illegal',
            n_roi_splits=1,
            roi_split=0,
            return_format='full',
        )
    assert 'Illegal value for train_mode!' in str(value_error)

    # Case 2
    with pytest.raises(ValueError) as value_error:
        train_all_models(
            roi_mask=roi_mask,
            stack=stack,
            parang=parang,
            obscon_array=obscon_array,
            selection_mask_config=selection_mask_config,
            base_model_creator=base_model_creator,
            n_signal_times=3,
            psf_template=psf_template,
            n_train_splits=2,
            train_mode='default',
            n_roi_splits=1,
            roi_split=0,
            return_format='illegal',
        )
    assert 'Illegal value for return_format!' in str(value_error)

    # Case 3
    with pytest.raises(ValueError) as value_error:
        train_all_models(
            roi_mask=roi_mask,
            stack=stack,
            parang=parang,
            obscon_array=obscon_array,
            selection_mask_config=selection_mask_config,
            base_model_creator=base_model_creator,
            n_signal_times=3,
            psf_template=psf_template,
            n_train_splits=2,
            train_mode='default',
            n_roi_splits=-2,
            roi_split=0,
            return_format='full',
        )
    assert 'n_roi_splits must be a positive integer!' in str(value_error)

    # Case 4
    with pytest.raises(ValueError) as value_error:
        train_all_models(
            roi_mask=roi_mask,
            stack=stack,
            parang=parang,
            obscon_array=obscon_array,
            selection_mask_config=selection_mask_config,
            base_model_creator=base_model_creator,
            n_signal_times=3,
            psf_template=psf_template,
            n_train_splits=2,
            train_mode='default',
            n_roi_splits=2,
            roi_split=4,
            return_format='full',
        )
    assert 'roi_split must be an integer in [0,' in str(value_error)

    # Case 5
    results = train_all_models(
        roi_mask=roi_mask,
        stack=stack,
        parang=parang,
        obscon_array=obscon_array,
        selection_mask_config=selection_mask_config,
        base_model_creator=base_model_creator,
        n_signal_times=3,
        psf_template=psf_template,
        n_train_splits=2,
        train_mode='signal_masking',
        n_roi_splits=1,
        roi_split=0,
        return_format='full',
    )
    assert tuple(results['stack_shape']) == stack.shape
    assert np.isclose(np.nansum(results['residuals']['default']), 0)

    # Case 6
    results = train_all_models(
        roi_mask=roi_mask,
        stack=stack,
        parang=parang,
        obscon_array=obscon_array,
        selection_mask_config=selection_mask_config,
        base_model_creator=base_model_creator,
        n_signal_times=3,
        psf_template=psf_template,
        n_train_splits=2,
        train_mode='signal_masking',
        n_roi_splits=2,
        roi_split=0,
        return_format='partial',
    )
    assert np.isclose(np.nansum(results['residuals']['default']), 0)
