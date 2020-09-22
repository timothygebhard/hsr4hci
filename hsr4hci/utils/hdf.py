"""
Utility functions for dealing with HDF files.
The code here is based on: https://codereview.stackexchange.com/a/121308
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Union

import h5py
import numpy as np


# -----------------------------------------------------------------------------
# CONSTANT DEFINITIONS
# -----------------------------------------------------------------------------

# Define a (incomplete) list of types that h5py supports for reading and
# writing. Note that each type is mapped to a native numpy type by h5py.
H5PY_SUPPORTED_TYPES = (
    bool,
    bytes,
    complex,
    float,
    int,
    np.complex,
    np.float,
    np.generic,
    np.int,
    np.ndarray,
    np.str,
    str,
)


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def save_dict_to_hdf(dictionary: dict, file_path: Union[Path, str]) -> None:
    """
    Save the given `dictionary` as an HDF file at the given `file_path`.
    If the `dictionary` is nested, the HDF file will replicate this
    structure using groups.

    Args:
        dictionary: A (possibly nested) dictionary to be saved.
        file_path: The path to the target file (including name and
            file ending).
    """

    # Make sure that file_path is a proper Path
    file_path = Path(file_path)

    # Open an HDF file at the given location
    with h5py.File(file_path, 'w') as hdf_file:

        # Recursively loop over the given dictionary and store its contents
        recursively_save_dict_contents_to_group(
            hdf_object=hdf_file, prefix='/', dictionary=dictionary
        )


def recursively_save_dict_contents_to_group(
    hdf_object: Union[h5py.File, h5py.Group], prefix: str, dictionary: dict
) -> None:
    """
    Auxiliary function for recursively looping over the contents of a
    dictionary and saving them to an HDF file.

    Args:
        hdf_object: Either an open HDF file, or a a group inside such
            a file.
        prefix: Path to the location inside the HDF file; e.g., the
            name of a group, or a path (for nested groups).
        dictionary: The dictionary to be saved at the given location.
    """

    # Loop over the given dictionary
    for key, item in dictionary.items():

        # Define the path where the current key should be stored
        path = f'{prefix}/{key}'

        # If the current `item` is a dict, we have to create a group in the
        # file by calling this method recursively on `item`
        if isinstance(item, dict):
            recursively_save_dict_contents_to_group(
                hdf_object=hdf_object, prefix=path, dictionary=item
            )

        # If the current `item` contains data, create a dataset to store them
        elif isinstance(item, H5PY_SUPPORTED_TYPES):
            hdf_object[path] = item

        # If the type of `item` is not supported, raise a TypeError
        else:
            raise TypeError(f'Unsupported type {type(item)} for {path}!')


def load_dict_from_hdf(file_path: Union[Path, str]) -> dict:
    """
    Load the contents of an HDF file into a dictionary to replicate the
    internal structure (group, subgroups, ...) of the HDF file.

    Args:
        file_path: The path to the target HDF file.

    Returns:
        A `dict` containing the contents of the specified HDF file.
    """

    # Make sure that file_path is a proper Path
    file_path = Path(file_path)

    # Open the target HDF file
    with h5py.File(file_path, 'r') as hdf_file:

        # Recursively loop over its contents to load them into a dict
        return recursively_load_dict_contents_from_group(
            hdf_object=hdf_file, path='/'
        )


def recursively_load_dict_contents_from_group(
    hdf_object: Union[h5py.File, h5py.Group], path: str = ''
) -> dict:
    """
    Auxiliary function for recursively looping over the contents of a
    given `hdf_object` and loading them into a dictionary.

    Args:
        hdf_object: A HDF object; either an HDF file (root) or a group.
        path: The path to the `hdf_object` in the actual HDF file.

    Returns:
        The contents of `hdf_object[path]` as a dictionary.
    """

    # Initialize the output dict
    results = {}

    # Loop over the contents of the group (or root) at the current `path`
    for key, item in hdf_object[path].items():

        # If the current item is a dataset, load its value. h5py will
        # automatically convert it to a numpy type.
        if isinstance(item, h5py.Dataset):
            results[key] = item[()]

        # If the current item is a group, we recursively call this method on it
        elif isinstance(item, h5py.Group):
            new_path = f'{path}/{key}'
            results[key] = recursively_load_dict_contents_from_group(
                hdf_object=hdf_object, path=new_path,
            )

    return results
