"""
Tests for config.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path

import json

from _pytest.monkeypatch import MonkeyPatch
from _pytest.tmpdir import TempPathFactory
from deepdiff import DeepDiff

import pytest

from hsr4hci.config import (
    get_datasets_dir,
    get_experiments_dir,
    get_hsr4hci_dir,
    load_config,
)


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__load_config(tmp_path_factory: TempPathFactory) -> None:
    """
    Test `hsr4hci.config.load_config`.
    """

    # Define location of test config file in temporary directory
    test_dir = tmp_path_factory.mktemp('config', numbered=False)
    file_path = test_dir / 'test_config.json'

    # Create dummy test configuration
    test_config = {'a': 'a', 'b': 2, 'c': [1.0, 2.0, 3.0], 'd': {'x': 'y'}}

    # Create test config file
    with open(file_path, 'w') as json_file:
        json.dump(test_config, json_file, indent=2)

    # Test case 1
    config = load_config(file_path)
    deepdiff = DeepDiff(t1=config, t2=test_config)
    assert not deepdiff

    # Test case 2
    file_path = test_dir / 'this_file_should_not_exist'
    with pytest.raises(FileNotFoundError) as error:
        load_config(file_path)
    assert 'does not exist' in str(error)

    # Test case 3
    file_path_str = (test_dir / 'test_config.json').as_posix()
    config = load_config(file_path_str)
    deepdiff = DeepDiff(t1=config, t2=test_config)
    assert not deepdiff


def test__get_hsr4hci_dir() -> None:
    """
    Test `hsr4hci.config.get_hsr4hci_dir`.
    """

    assert get_hsr4hci_dir().exists()


def test__get_datasets_dir(monkeypatch: MonkeyPatch) -> None:
    """
    Test `hsr4hci.config.get_datasets_dir`.
    """

    # Case 1
    # Monkeypatch HSR4HCI_DATASETS_DIR to something that exists
    monkeypatch.setenv(
        'HSR4HCI_DATASETS_DIR',
        Path(__file__).parent.resolve().as_posix(),
    )
    assert get_datasets_dir().exists()

    # Case 2
    # Monkeypatch HSR4HCI_DATASETS_DIR to something that does not exist
    monkeypatch.setenv(
        'HSR4HCI_DATASETS_DIR',
        (Path(__file__).parent.resolve() / 'this_does_not_exist').as_posix(),
    )
    with pytest.raises(NotADirectoryError) as not_a_directory_error:
        get_datasets_dir()
    assert 'does not exist' in str(not_a_directory_error)

    # Case 3
    # Monkeypatch HSR4HCI_DATASETS_DIR to ensure that it is *NOT* set
    monkeypatch.delenv('HSR4HCI_DATASETS_DIR')
    with pytest.raises(KeyError) as key_error:
        get_datasets_dir()
    assert 'HSR4HCI_DATASETS_DIR not defined' in str(key_error)


def test__get_experiments_dir(monkeypatch: MonkeyPatch) -> None:
    """
    Test `hsr4hci.config.get_experiments_dir`.
    """

    # Case 1
    # Monkeypatch HSR4HCI_EXPERIMENTS_DIR to ensure that it is set
    monkeypatch.setenv(
        'HSR4HCI_EXPERIMENTS_DIR',
        Path(__file__).parent.resolve().as_posix(),
    )
    assert get_experiments_dir().exists()

    # Case 2
    # Monkeypatch HSR4HCI_DATASETS_DIR to something that does not exist
    monkeypatch.setenv(
        'HSR4HCI_EXPERIMENTS_DIR',
        (Path(__file__).parent.resolve() / 'this_does_not_exist').as_posix(),
    )
    with pytest.raises(NotADirectoryError) as not_a_directory_error:
        get_experiments_dir()
    assert 'does not exist' in str(not_a_directory_error)

    # Case 3
    # Monkeypatch HSR4HCI_EXPERIMENTS_DIR to ensure that it is not set
    monkeypatch.delenv('HSR4HCI_EXPERIMENTS_DIR')
    with pytest.raises(KeyError) as key_error:
        get_experiments_dir()
    assert 'HSR4HCI_EXPERIMENTS_DIR not defined' in str(key_error)
