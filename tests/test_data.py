"""
Tests for data.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Any

from _pytest.tmpdir import TempPathFactory
from deepdiff import DeepDiff

import h5py
import pytest
import numpy as np

# noinspection PyProtectedMember
from hsr4hci.data import (
    _resolve_name_or_path,
    load_dataset,
    load_metadata,
    load_observing_conditions,
    load_parang,
    load_planets,
    load_psf_template,
    load_stack,
)


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_data() -> Any:

    stack = np.random.normal(0, 1, (100, 51, 51)).astype(np.float32)
    parang = np.random.normal(0, 1, (100,)).astype(np.float32)
    psf_template = np.random.normal(0, 1, (33, 33)).astype(np.float32)
    observing_conditions = {
        'air_pressure': np.random.normal(0, 1, (100,)).astype(np.float32),
        'wind_speed_u': np.random.normal(0, 1, (100,)).astype(np.float32),
    }
    metadata = {
        'INSTRUMENT': 'NACO',
        'DIT_STACK': 0,
        'DIT_PSF_TEMPLATE': 0.2,
        'PIXSCALE': 0.0271,
    }
    planets = {
        "b": {"separation": 0.1867, "position_angle": 132.0, "contrast": 6.48}
    }

    return stack, parang, psf_template, observing_conditions, metadata, planets


@pytest.fixture(scope="session")
def test_file(tmp_path_factory: TempPathFactory, test_data: Any) -> Path:

    # Unpack the test data
    (
        stack,
        parang,
        psf_template,
        observing_conditions,
        metadata,
        planets,
    ) = test_data

    # Define location of test file in temporary directory
    test_dir = tmp_path_factory.mktemp('data', numbered=False)
    file_path = test_dir / 'test.hdf'

    # Write the test data to a HDF file that has the structure of a data set
    with h5py.File(file_path, 'w') as hdf_file:

        hdf_file.create_dataset(name='stack', data=stack)
        hdf_file.create_dataset(name='parang', data=parang)
        hdf_file.create_dataset(name='psf_template', data=psf_template)

        group = hdf_file.create_group(name='metadata')
        for key, value in metadata.items():
            group.create_dataset(name=key, data=value)

        group = hdf_file.create_group(name='observing_conditions/interpolated')
        for key, value in observing_conditions.items():
            group.create_dataset(name=key, data=value)

        group = hdf_file.create_group(name='planets/b')
        for key, value in planets['b'].items():
            group.create_dataset(name=key, data=value)

    return file_path


def test__resolve_name_or_path() -> None:

    # Case 1
    file_path = _resolve_name_or_path(name_or_path='test')
    assert file_path.name == 'test.hdf'

    # Case 2
    name_or_path = Path('path', 'to', 'file.hdf')
    file_path = _resolve_name_or_path(name_or_path=name_or_path)
    assert file_path.name == 'file.hdf'

    # Case 3
    with pytest.raises(ValueError) as value_error:
        # noinspection PyTypeChecker
        _resolve_name_or_path(name_or_path=5)  # type: ignore
    assert 'name_or_path must be a string or a Path!' in str(value_error)


def test_load_parang(test_file: Path, test_data: Any) -> None:

    _, parang_1, _, _, _, _ = test_data
    parang_2 = load_parang(name_or_path=test_file)
    assert np.allclose(parang_1, parang_2)


def test_load_psf_template(test_file: Path, test_data: Any) -> None:

    # Case 1
    _, _, psf_template_1, _, _, _ = test_data
    psf_template_2 = load_psf_template(name_or_path=test_file)
    assert np.allclose(psf_template_1, psf_template_2)

    # Case 2
    test_dir = test_file.parent
    file_path = test_dir / 'psf_template.hdf'
    with h5py.File(file_path, 'w') as hdf_file:
        hdf_file.create_dataset(
            name='psf_template',
            data=np.random.normal(0, 1, (3, 33, 33))
        )
    with pytest.raises(RuntimeError) as runtime_error:
        load_psf_template(name_or_path=file_path)
    assert 'psf_template is not 2D!' in str(runtime_error)


def test_load_observing_conditions(test_file: Path, test_data: Any) -> None:

    _, _, _, observing_conditions_1, _, _ = test_data
    observing_conditions_2 = load_observing_conditions(name_or_path=test_file)
    deepdiff = DeepDiff(
        t1=observing_conditions_1,
        t2=observing_conditions_2.as_dict('all'),
        ignore_type_in_groups=[(float, np.float64), (int, np.int64)],
        ignore_nan_inequality=True,
    )
    assert not deepdiff


def test_load_metadata(test_file: Path, test_data: Any) -> None:

    _, _, _, _, metadata_1, _ = test_data
    metadata_2 = load_metadata(name_or_path=test_file)
    deepdiff = DeepDiff(
        t1=metadata_1,
        t2=metadata_2,
        ignore_type_in_groups=[(float, np.float64), (int, np.int64)],
        ignore_nan_inequality=True,
    )
    assert not deepdiff


def test_load_planets(test_file: Path, test_data: Any) -> None:

    _, _, _, _, _, planets_1 = test_data
    planets_2 = load_planets(name_or_path=test_file)
    deepdiff = DeepDiff(
        t1=planets_1,
        t2=planets_2,
        ignore_type_in_groups=[(float, np.float64), (int, np.int64)],
        ignore_nan_inequality=True,
    )
    assert not deepdiff


def test_stack(test_file: Path, test_data: Any) -> None:

    stack_1, _, _, _, _, _ = test_data

    # Case 1
    stack_2 = load_stack(name_or_path=test_file)
    assert np.allclose(stack_1, stack_2)

    # Case 2
    stack_3 = load_stack(name_or_path=test_file, remove_planets=True)
    assert np.allclose(stack_1, stack_3)

    # Case 3
    stack_4 = load_stack(name_or_path=test_file, frame_size=(3, 3))
    assert np.allclose(stack_1[:, 24:27, 24:27], stack_4)

    # Case 4
    stack_4 = load_stack(name_or_path=test_file, frame_size=(53, 53))
    assert np.allclose(stack_1, stack_4)


def test_load_dataset(test_file: Path, test_data: Any) -> None:

    (
        stack_1,
        parang_1,
        psf_template_1,
        observing_conditions_1,
        metadata_1,
        _
    ) = test_data
    (
        stack_2,
        parang_2,
        psf_template_2,
        observing_conditions_2,
        metadata_2,
    ) = load_dataset(name_or_path=test_file)
    assert np.allclose(stack_1, stack_2)
    assert np.allclose(parang_1, parang_2)
    assert np.allclose(psf_template_1, psf_template_2)
    assert not DeepDiff(
        t1=observing_conditions_1,
        t2=observing_conditions_2.as_dict('all'),
        ignore_type_in_groups=[(float, np.float64), (int, np.int64)],
        ignore_nan_inequality=True,
    )
    assert not DeepDiff(
        t1=metadata_1,
        t2=metadata_2,
        ignore_type_in_groups=[(float, np.float64), (int, np.int64)],
        ignore_nan_inequality=True,
    )
