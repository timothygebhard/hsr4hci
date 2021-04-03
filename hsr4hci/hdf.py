"""
Utility functions for dealing with HDF files.

Parts of the code in the module are based on:
    https://codereview.stackexchange.com/a/121308
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Union

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
    np.generic,
    np.ndarray,
    str,
)


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def is_hdf_file(file_path: Union[Path, str]) -> bool:
    """
    Check the ending of a file_path to determine if it is an HDF file.

    Args:
        file_path: Path to the target file to be checked.

    Returns:
        True if the given file is a HDF file; False otherwise.
    """

    endswith_hdf = Path(file_path).name.endswith('.hdf')
    endswith_hdf5 = Path(file_path).name.endswith('.hdf5')

    return endswith_hdf or endswith_hdf5


def save_data_to_hdf(
    hdf_file: h5py.File,
    location: str,
    name: str,
    data: Any,
    overwrite: bool = True,
) -> None:
    """
    Auxiliary function to write data to an open HDF file that provides
    automatic overwriting (which requires deleting and re-creating data
    sets that already exist).

    Args:
        hdf_file: An open HDF file (in write mode).
        location: The path ("group_1/group_2/.../group_n") at which to
            create the new data set in the HDF file. Can be empty.
        name: The name of the data set.
        data: The data to be written to the data set.
        overwrite: Whether or not to overwrite a data set of the same
            name that already exists in the given location.
    """

    # Ensure that we only try to save supported types
    if not isinstance(data, H5PY_SUPPORTED_TYPES):
        raise ValueError(f'Type "{type(data)}" not supported by HDF format!')

    # Check if the data set already exists
    if (location in hdf_file) and (name in hdf_file[location]):

        # If overwrite is True, we delete the data set and create it again
        # below (there is no direct overwrite)
        if overwrite:
            del hdf_file[location][name]

        # Otherwise, we raise an error
        else:
            raise KeyError(f'Data set with name "{name}" already exists!')

    # Finally, we create the full path and store it. Groups are automatically
    # created as needed by h5py.
    full_path = location.strip('/') + '/' + name.strip('/')
    hdf_file.create_dataset(name=full_path, data=data)


def save_dict_to_hdf(
    dictionary: dict,
    file_path: Union[Path, str],
    mode: str = 'a',
    prefix: str = '',
) -> None:
    """
    Save the given `dictionary` as an HDF file at the given `file_path`.
    If the `dictionary` is nested, the HDF file will replicate this
    structure using groups.

    Args:
        dictionary: A (possibly nested) dictionary to be saved.
        file_path: The path to the target file (including name and
            file ending).
        mode: The mode (e.g., "w" or "a") that is used when opening
            the HDF file for writing.
        prefix: Prefix to use when writing to the HDF file. This can
            be used, for example, to write the dictionary into its own
            group inside the HDF file.
    """

    # Make sure that file_path is a proper Path
    file_path = Path(file_path)

    # Open an HDF file at the given location
    with h5py.File(file_path, mode=mode) as hdf_file:

        # Recursively loop over the given dictionary and store its contents
        recursively_save_dict_contents_to_group(
            hdf_object=hdf_file, prefix=prefix, dictionary=dictionary
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

        # If the current `item` contains data, create a dataset to store them.
        # If the data set already exists, delete it (overwriting existing data
        # sets is not possible otherwise).
        elif isinstance(item, H5PY_SUPPORTED_TYPES):
            if path in hdf_object:
                del hdf_object[path]
            hdf_object.create_dataset(name=path, data=item)

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
            value = item[()]
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            results[key] = value

        # If the current item is a group, we recursively call this method on it
        elif isinstance(item, h5py.Group):
            new_path = f'{path}/{key}'
            results[key] = recursively_load_dict_contents_from_group(
                hdf_object=hdf_object,
                path=new_path,
            )

    return results
