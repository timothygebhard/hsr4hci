"""
Tests for observing_conditions.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import List, Tuple

from _pytest.tmpdir import TempPathFactory
from deepdiff import DeepDiff

import numpy as np
import pytest

from hsr4hci.fits import save_fits
from hsr4hci.hdf import save_dict_to_hdf
from hsr4hci.merging import (
    get_list_of_fits_file_paths,
    get_list_of_hdf_file_paths,
    merge_fits_files,
    merge_hdf_files,
)


# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def hdf_dir(tmp_path_factory: TempPathFactory) -> Path:
    return tmp_path_factory.mktemp('merging_hdf', numbered=False)


@pytest.fixture(scope="session")
def fits_dir(tmp_path_factory: TempPathFactory) -> Path:
    return tmp_path_factory.mktemp('merging_fits', numbered=False)


@pytest.fixture(scope="session")
def fits_data(fits_dir: Path) -> Tuple[Path, np.ndarray, List[Path]]:

    np.random.seed(42)

    # Create some test data
    n_files = 10
    file_paths = []
    array = np.random.normal(0, 1, (101, 101)).astype(np.float32)
    mask = np.random.randint(0, n_files, (101, 101))
    for i in range(n_files):
        file_path = fits_dir / f'dummy_{i + 1}-{n_files}.fits'
        file_paths.append(file_path)
        tmp_mask = mask == i
        tmp_array = np.copy(array)
        tmp_array[~tmp_mask] = np.nan
        save_fits(array=tmp_array, file_path=file_path)

    return fits_dir, array, sorted(file_paths)


@pytest.fixture(scope="session")
def hdf_data__partial(fits_dir: Path) -> Tuple[Path, dict, List[Path]]:

    np.random.seed(42)

    # Create some test data
    n_files = 10
    stack_shape = (17, 51, 51)
    file_paths = []
    array_default = np.random.normal(0, 1, stack_shape).astype(np.float32)
    array_0 = np.random.normal(0, 1, stack_shape).astype(np.float32)
    array_1 = np.random.normal(0, 1, stack_shape).astype(np.float32)
    array_2 = np.random.normal(0, 1, stack_shape).astype(np.float32)
    mask = np.random.randint(0, n_files, stack_shape[1:])
    for i in range(n_files):
        file_path = fits_dir / f'residuals_{i + 1}-{n_files}.hdf'
        file_paths.append(file_path)
        tmp_mask = mask == i
        residuals = {
            'stack_shape': np.array([17, 51, 51]),
            'roi_mask': tmp_mask,
            'residuals': {
                'default': array_default[:, tmp_mask],
                '0': array_0[:, tmp_mask],
                '1': array_1[:, tmp_mask],
                '2': array_2[:, tmp_mask],
            },
        }
        save_dict_to_hdf(dictionary=residuals, file_path=file_path)

    full_residuals = {
        'default': array_default,
        '0': array_0,
        '1': array_1,
        '2': array_2,
    }

    return fits_dir, full_residuals, sorted(file_paths)


@pytest.fixture(scope="session")
def hdf_data__full(fits_dir: Path) -> Tuple[Path, dict, List[Path]]:

    np.random.seed(42)

    # Create some test data
    n_files = 10
    stack_shape = (17, 51, 51)
    file_paths = []
    array_default = np.random.normal(0, 1, stack_shape).astype(np.float32)
    array_0 = np.random.normal(0, 1, stack_shape).astype(np.float32)
    array_1 = np.random.normal(0, 1, stack_shape).astype(np.float32)
    array_2 = np.random.normal(0, 1, stack_shape).astype(np.float32)
    mask = np.random.randint(0, n_files, stack_shape[1:])
    for i in range(n_files):
        file_path = fits_dir / f'residuals_{i + 1}-{n_files}.hdf'
        file_paths.append(file_path)
        tmp_mask = mask == i
        tmp_array_default = np.copy(array_default)
        tmp_array_default[:, ~tmp_mask] = np.nan
        tmp_array_0 = np.copy(array_0)
        tmp_array_0[:, ~tmp_mask] = np.nan
        tmp_array_1 = np.copy(array_1)
        tmp_array_1[:, ~tmp_mask] = np.nan
        tmp_array_2 = np.copy(array_2)
        tmp_array_2[:, ~tmp_mask] = np.nan
        residuals = {
            'stack_shape': np.array([17, 51, 51]),
            'roi_mask': tmp_mask,
            'residuals': {
                'default': tmp_array_default,
                '0': tmp_array_0,
                '1': tmp_array_1,
                '2': tmp_array_2,
            },
        }
        save_dict_to_hdf(dictionary=residuals, file_path=file_path)

    full_residuals = {
        'default': array_default,
        '0': array_0,
        '1': array_1,
        '2': array_2,
    }

    return fits_dir, full_residuals, sorted(file_paths)


def test__get_list_of_fits_file_paths(
    fits_data: Tuple[Path, np.ndarray, List[Path]]
) -> None:

    fits_dir, array, expected_file_paths = fits_data

    # Case 1
    actual_file_paths = get_list_of_fits_file_paths(fits_dir, 'dummy')
    assert actual_file_paths == expected_file_paths

    # Case 2
    file_path = fits_dir / 'dummy_unexpected.fits'
    file_path.touch()
    with pytest.warns(UserWarning, match='(Naming convention suggests).*'):
        get_list_of_fits_file_paths(fits_dir, 'dummy')
    file_path.unlink()
    assert not file_path.exists()


def test__merge_fits_files(
    fits_data: Tuple[Path, np.ndarray, List[Path]]
) -> None:

    _, array, file_paths = fits_data

    # Case 1
    merged = merge_fits_files(fits_file_paths=file_paths)
    assert np.allclose(array, merged)


def test__get_list_of_hdf_file_paths(
    hdf_data__partial: Tuple[Path, np.ndarray, List[Path]]
) -> None:

    hdf_dir, full_results, expected_file_paths = hdf_data__partial

    # Case 1
    actual_file_paths = get_list_of_hdf_file_paths(hdf_dir)
    assert actual_file_paths == expected_file_paths

    # Case 2
    file_path = hdf_dir / 'residuals_unexpected.hdf'
    file_path.touch()
    with pytest.warns(UserWarning, match='(Naming convention suggests).*'):
        get_list_of_hdf_file_paths(hdf_dir)
    file_path.unlink()
    assert not file_path.exists()


def test__merge_hdf_files__partial(
    hdf_data__partial: Tuple[Path, np.ndarray, List[Path]]
) -> None:

    hdf_dir, full_results, file_paths = hdf_data__partial

    # Case 1
    merged = merge_hdf_files(hdf_file_paths=file_paths)
    deepdiff = DeepDiff(
        t1=full_results,
        t2=merged,
        ignore_type_in_groups=[(float, np.float64)],
        ignore_nan_inequality=True,
    )
    assert not deepdiff


def test__merge_hdf_files__full(
    hdf_data__full: Tuple[Path, np.ndarray, List[Path]]
) -> None:

    hdf_dir, full_results, file_paths = hdf_data__full

    # Case 1
    merged = merge_hdf_files(hdf_file_paths=file_paths)
    deepdiff = DeepDiff(
        t1=full_results,
        t2=merged,
        ignore_type_in_groups=[(float, np.float64)],
        ignore_nan_inequality=True,
    )
    assert not deepdiff
