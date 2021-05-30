"""
Tests for hdf.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import os

from _pytest.tmpdir import TempPathFactory
from deepdiff import DeepDiff

import h5py
import numpy as np
import pytest

from hsr4hci.hdf import (
    create_hdf_dir,
    load_dict_from_hdf,
    save_data_to_hdf,
    save_dict_to_hdf,
)


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def hdf_dir(tmp_path_factory: TempPathFactory) -> Path:
    return tmp_path_factory.mktemp('hdf', numbered=False)


@pytest.fixture(scope="session")
def test_data() -> dict:
    return {
        'bool': False,
        'complex': 1 + 2j,
        'float': 3.0,
        'int': 2,
        'array': np.array([0.0, 1.0]),
        'dict': {'a': 'string', 'b': np.array([1, 2, 3, 4, 5])},
        'double_dict': {
            'dict_1': {'1': 'a', '2': 123},
            'dict_2': {'nan': np.nan, 'empty': np.empty((42, 23))},
        },
    }


def test__save_data_to_hdf(hdf_dir: Path) -> None:

    data_1 = np.random.normal(0, 1, 10)
    data_2 = np.random.normal(1, 2, 3)

    # Case 1
    file_path = hdf_dir / 'test_1.hdf'
    with h5py.File(file_path, 'w') as hdf_file:
        with pytest.raises(TypeError) as type_error:
            save_data_to_hdf(
                hdf_file=hdf_file,
                location='test_group',
                name='test_dataset',
                data=Path('.'),
                overwrite=True,
            )
        assert 'not supported by HDF format' in str(type_error)

    # Case 2
    file_path = hdf_dir / 'test_1.hdf'
    with h5py.File(file_path, 'w') as hdf_file:
        save_data_to_hdf(
            hdf_file=hdf_file,
            location='test_group',
            name='test_dataset',
            data=data_1,
            overwrite=True,
        )
        assert np.array_equal(hdf_file['test_group']['test_dataset'], data_1)

    # Case 3 + 4
    file_path = hdf_dir / 'test_1.hdf'
    with h5py.File(file_path, 'a') as hdf_file:

        # Case 3
        with pytest.raises(KeyError) as key_error:
            save_data_to_hdf(
                hdf_file=hdf_file,
                location='test_group',
                name='test_dataset',
                data=data_2,
                overwrite=False,
            )
        assert 'Data set with name' in str(key_error)

        # Case 4
        save_data_to_hdf(
            hdf_file=hdf_file,
            location='test_group',
            name='test_dataset',
            data=data_2,
            overwrite=True,
        )
        assert np.array_equal(hdf_file['test_group']['test_dataset'], data_2)


def test__save_dict_to_hdf(
    tmp_path_factory: TempPathFactory, hdf_dir: Path, test_data: dict
) -> None:

    # Define location of test file in temporary directory
    file_path = hdf_dir / 'test_2.hdf'

    # Case 1
    save_dict_to_hdf(dictionary=test_data, file_path=file_path)

    # Case 2 (repeat case 1 to check if overwrites work)
    save_dict_to_hdf(dictionary=test_data, file_path=file_path)

    # Case 3
    with pytest.raises(TypeError) as type_error:
        save_dict_to_hdf(dictionary={'path': Path('.')}, file_path=file_path)
    assert 'Unsupported type' in str(type_error)


def test__load_dict_from_hdf(
    tmp_path_factory: TempPathFactory, hdf_dir: Path, test_data: dict
) -> None:

    # Define location of test file in temporary directory
    file_path = hdf_dir / 'test_2.hdf'

    # Read the test HDF file
    data = load_dict_from_hdf(file_path=file_path)

    # Compute the difference between the original test data and the data that
    # we have just loaded from the test HDF file. We need to ignore various
    # type changes, because h5py automatically casts built-in Python types to
    # their corresponding numpy types.
    deepdiff = DeepDiff(
        t1=test_data,
        t2=data,
        ignore_type_in_groups=[
            (bool, np.bool_),
            (int, np.int64),
            (float, np.float64),
            (complex, np.complex128),
        ],
        ignore_nan_inequality=True,
    )

    # Make sure that nothing has changed
    assert not deepdiff


def test__create_hdf_dir(hdf_dir: Path) -> None:

    # Case 1
    test_dir = create_hdf_dir(experiment_dir=hdf_dir, create_on_work=False)
    assert test_dir.exists()
    assert not os.listdir(test_dir)

    # Case 2
    file_path = test_dir / 'dummy.hdf'
    file_path.touch()
    dir_path = test_dir / 'dummy_dir'
    dir_path.mkdir()
    test_dir = create_hdf_dir(experiment_dir=hdf_dir, create_on_work=False)
    assert test_dir.exists()
    assert not os.listdir(test_dir)
