"""
Tests for hdf.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from _pytest.tmpdir import TempPathFactory
from deepdiff import DeepDiff

import numpy as np

from hsr4hci.utils.hdf import save_dict_to_hdf, load_dict_from_hdf


# -----------------------------------------------------------------------------
# CONSTANT DEFINITIONS
# -----------------------------------------------------------------------------

# Define test dictionary to be saved to and loaded from HDF
TEST_DICT = {
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


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__save_dict_to_hdf(tmp_path_factory: TempPathFactory) -> None:

    # Define location of test file in temporary directory
    test_dir = tmp_path_factory.mktemp('hdf', numbered=False)
    file_path = test_dir / 'test.hdf'

    # Save to HDF; this test succeeds if no error occurs here
    save_dict_to_hdf(dictionary=TEST_DICT, file_path=file_path)


def test__load_dict_from_hdf(tmp_path_factory: TempPathFactory) -> None:

    # Define location of test file in temporary directory
    test_dir = tmp_path_factory.getbasetemp() / 'hdf'
    file_path = test_dir / 'test.hdf'

    # Read the test HDF file
    data = load_dict_from_hdf(file_path=file_path)

    # Compute the difference between the original test data and the data that
    # we have just loaded from the test HDF file. We need to ignore various
    # type changes, because h5py automatically casts built-in Python types to
    # their corresponding numpy types.
    deepdiff = DeepDiff(
        t1=TEST_DICT,
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
