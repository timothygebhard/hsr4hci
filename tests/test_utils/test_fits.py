"""
Tests for fits.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from _pytest.tmpdir import TempPathFactory

from hsr4hci.utils.fits import read_fits, save_fits


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__save_fits(tmp_path_factory: TempPathFactory) -> None:

    # Define location of test file in temporary directory
    test_dir = tmp_path_factory.mktemp('fits', numbered=False)
    file_path = test_dir / 'test.fits'

    # Define contents of FITS file
    array = np.arange(30).reshape((2, 3, 5))
    header = dict(str_key='Test Name',
                  int_key=5,
                  float_key=8.2,
                  nan_key=np.nan,
                  array_key=np.array([1, 2, 3]),
                  nan_array_key=np.array([1, 2, np.nan]))

    # Save to FITS
    save_fits(array=array, file_path=str(file_path), header=header)


def test__read_fits(tmp_path_factory: TempPathFactory) -> None:

    # Define location of test file in temporary directory
    test_dir = tmp_path_factory.getbasetemp() / 'fits'
    file_path = test_dir / 'test.fits'

    # Read the test FITS file
    array, header = read_fits(file_path=str(file_path), return_header=True)

    # Define the expected file contents, including the various expected
    # changes (e.g., serialization of values, capitalization of keys) that
    # are expected to happen when saving a header to a FITS file
    expected_array = np.arange(30).reshape((2, 3, 5))
    expected_header = dict(STR_KEY='Test Name',
                           INT_KEY=5,
                           FLOAT_KEY=8.2,
                           NAN_KEY='NaN',
                           ARRAY_KEY='[1, 2, 3]',
                           NAN_ARRAY_KEY='[1.0, 2.0, NaN]')

    for key, value in expected_header.items():
        assert header[key] == expected_header[key]
    assert np.array_equal(array, expected_array)
