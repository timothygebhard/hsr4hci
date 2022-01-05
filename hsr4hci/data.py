"""
Utility functions for loading data sets.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from astropy.units import Quantity

import h5py
import numpy as np

from hsr4hci.config import get_datasets_dir
from hsr4hci.forward_modeling import add_fake_planet
from hsr4hci.general import prestack_array
from hsr4hci.observing_conditions import ObservingConditions


# -----------------------------------------------------------------------------
# AUXILIARY FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def _resolve_name_or_path(name_or_path: Union[str, Path]) -> Path:
    """
    Resolve a given `name_or_path` to the file path of an HDF file.

    Args:
        name_or_path: Either a string or a Path. In case it is a string,
            it is assumed it is the name of a data set in the hsr4hci
            `datasets` directory, and the corresponding file path is
            constructed. In case it is a Path (or a string that contains
            a slash), it is assumed that this is the path to the target
            HDF file, so we return it without any modifications. In all
            other cases, an error is raised.

    Returns:
        File path to an HDF file containing a data set.
    """

    # Define shortcuts for conditions
    is_path = isinstance(name_or_path, Path)
    is_string = isinstance(name_or_path, str)
    has_slash = False if is_path else ('/' in str(name_or_path))

    # If `name_or_path` is a string that does not contain any slashes, we
    # assume it to be the name of a data set, and we can resolve it to the
    # (expected) location of the data set
    if is_string and not has_slash:
        return (
            get_datasets_dir()
            / name_or_path
            / 'output'
            / f'{name_or_path}.hdf'
        )

    # If it is a Path (or a string with slashes), we assume that it is the
    # Path to the HDF file which contains the data set
    if is_path or (is_string and has_slash):
        return Path(name_or_path).expanduser()

    # In any other case, we raise an error
    raise ValueError('name_or_path must be a string or a Path!')


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def load_parang(
    name_or_path: Union[str, Path],
    binning_factor: int = 1,
    **_: Any,
) -> np.ndarray:
    """
    Load the parallactic angles.

    Args:
        name_or_path: Name of a data set (e.g., "beta_pictoris__lp"),
            or Path to the HDF file that contains a data set.
        binning_factor: Number of time steps that should be temporally
            binned ("pre-stacked") using a block-wise mean.

    Returns:
        A numpy array containing the parallactic angles.
    """

    # Get the path to the HDF file that contains the data to be loaded
    file_path = _resolve_name_or_path(name_or_path)

    # Read in the data set from the HDF file
    with h5py.File(file_path, 'r') as hdf_file:
        parang = np.array(hdf_file['parang'], dtype=np.float32)

    # Temporally bin the parallactic angles
    parang = prestack_array(array=parang, stacking_factor=binning_factor)

    return parang


def load_psf_template(name_or_path: Union[str, Path], **_: Any) -> np.ndarray:
    """
    Load the unsaturated PSF template.

    Args:
        name_or_path: Name of a data set (e.g., "beta_pictoris__lp"),
            or Path to an HDF file that contains the data set.

    Returns:
        A numpy array containing the unsaturated PSF template.
    """

    # Get the path to the HDF file that contains the data to be loaded
    file_path = _resolve_name_or_path(name_or_path)

    # Read in the data set from the HDF file
    with h5py.File(file_path, 'r') as hdf_file:
        psf_template = np.array(hdf_file['psf_template'], dtype=np.float32)
        psf_template = psf_template.squeeze()

    # Ensure that the PSF template is two-dimensional now; otherwise this
    # can result in weird errors that are hard to debug
    if psf_template.ndim != 2:
        raise RuntimeError(
            f'psf_template is not 2D! (shape = {psf_template.shape})'
        )

    return psf_template


def load_observing_conditions(
    name_or_path: Union[str, Path], binning_factor: int = 1, **_: Any
) -> ObservingConditions:
    """
    Load the observing conditions.

    Args:
        name_or_path: Name of a data set (e.g., "beta_pictoris__lp"),
            or Path to an HDF file that contains the data set.
        binning_factor: Number of time steps that should be temporally
            binned ("pre-stacked") using a block-wise mean.

    Returns:
        An `ObservingConditions` object containing the observing
        conditions.
    """

    # Get the path to the HDF file that contains the data to be loaded
    file_path = _resolve_name_or_path(name_or_path)

    # Read in the data set from the HDF file
    with h5py.File(file_path, 'r') as hdf_file:

        # Collect the observing conditions into a (temporary) dictionary
        _observing_conditions = dict()
        for key in hdf_file['observing_conditions']['interpolated'].keys():
            _observing_conditions[key] = np.array(
                hdf_file['observing_conditions']['interpolated'][key],
                dtype=np.float32,
            )

    # Apply temporal binning to the observing conditions
    for key in _observing_conditions.keys():
        _observing_conditions[key] = prestack_array(
            array=_observing_conditions[key], stacking_factor=binning_factor
        )

    # Convert the observing conditions into an ObservingConditions object
    observing_conditions = ObservingConditions(_observing_conditions)

    return observing_conditions


def load_metadata(name_or_path: Union[str, Path], **_: Any) -> dict:
    """
    Load the metadata.

    Args:
        name_or_path: Name of a data set (e.g., "beta_pictoris__lp"),
            or Path to an HDF file that contains the data set.

    Returns:
        A dictionary containing the metadata.
    """

    # Initialize metadata
    metadata: Dict[str, Union[str, float]] = dict()

    # Get the path to the HDF file that contains the data to be loaded
    file_path = _resolve_name_or_path(name_or_path)

    # Read in the data set from the HDF file
    with h5py.File(file_path, 'r') as hdf_file:
        for key in hdf_file['metadata'].keys():
            value = hdf_file['metadata'][key][()]
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            metadata[key] = value

    return metadata


def load_planets(name_or_path: Union[str, Path], **_: Any) -> dict:
    """
    Load information about the planets (i.e., positions and contrasts).

    Args:
        name_or_path: Name of a data set (e.g., "beta_pictoris__lp"),
            or Path to an HDF file that contains the data set.

    Returns:
        A dictionary containing the planet information.
    """

    # Initialize planet information
    planets: Dict[str, Dict[str, float]] = dict()

    # Get the path to the HDF file that contains the data to be loaded
    file_path = _resolve_name_or_path(name_or_path)

    # Read in the planet information from the HDF file
    with h5py.File(file_path, 'r') as hdf_file:
        for key in hdf_file['planets'].keys():
            planets[key] = dict(
                separation=hdf_file['planets'][key]['separation'][()],
                position_angle=hdf_file['planets'][key]['position_angle'][()],
                contrast=hdf_file['planets'][key]['contrast'][()],
            )

    return planets


def load_stack(
    name_or_path: Union[str, Path],
    binning_factor: int = 1,
    frame_size: Optional[Tuple[int, int]] = None,
    remove_planets: bool = False,
) -> np.ndarray:
    """
    Load the stack.

    Args:
        name_or_path: Name of a data set (e.g., "beta_pictoris__lp"),
            or Path to an HDF file that contains the data set.
        binning_factor: Number of frames that should be temporally
            binned ("pre-stacked") using a block-wise mean.
        frame_size: Target frame size to which the stack should be
            (spatially) cropped.
        remove_planets: If yes, negative fake planets are injected at
            the positions of the known planets to remove them from the
            stack. Useful for experiments with fake planets, for which
            a "clean" stack is required.

    Returns:
        A 3D numpy array of shape `(n_frames, x_size, y_size)` that
        contains the stack after cropping / binning / planet removal.
    """

    # Get the path to the HDF file that contains the data set to be loaded
    file_path = _resolve_name_or_path(name_or_path)

    # Open the HDF file to read the stack from it
    with h5py.File(file_path, 'r') as hdf_file:

        # Get shape of the stack as it is in the HDF file
        stack_shape = hdf_file['stack'].shape

        # Define target shape (= stack shape after spatial cropping)
        if frame_size is not None:
            target_shape = (-1, frame_size[0], frame_size[1])
        else:
            target_shape = (-1, -1, -1)

        # Compute slices that can be used to select a cropped version of the
        # stack directly (this is much faster than loading the entire stack
        # into memory to crop it there). The code here is basically the same
        # as the one in `hsr4hci.general.crop_center()`.
        slices = list()
        for old_len, new_len in zip(stack_shape, target_shape):
            if new_len > old_len:
                start = None
                end = None
            else:
                start = old_len // 2 - new_len // 2 if new_len != -1 else None
                end = start + new_len if start is not None else None
            slices.append(slice(start, end))

        # Finally, load only the cropped stack into a numpy array
        stack = np.array(hdf_file['stack'][tuple(slices)], dtype=np.float32)

    # Remove planets from stack, if desired
    if remove_planets:

        # Load parallactic angles, PSF template, metadata and information
        # about the planets from the HDF file
        parang = load_parang(name_or_path=name_or_path)
        psf_template = load_psf_template(name_or_path=name_or_path)
        metadata = load_metadata(name_or_path=name_or_path)
        planets = load_planets(name_or_path=name_or_path)

        # Define shortcut (PIXSCALE is in arcsec / pixel)
        pixscale = float(metadata['PIXSCALE'])

        # Loop over the existing planets and add negative fake planets at
        # their positions to remove them
        for name, parameters in planets.items():
            stack = np.asarray(
                add_fake_planet(
                    stack=stack,
                    parang=parang,
                    psf_template=psf_template,
                    polar_position=(
                        Quantity(parameters['separation'] / pixscale, 'pixel'),
                        Quantity(parameters['position_angle'], 'degree'),
                    ),
                    magnitude=float(parameters['contrast']),
                    extra_scaling=-1,
                    dit_stack=float(metadata['DIT_STACK']),
                    dit_psf_template=float(metadata['DIT_PSF_TEMPLATE']),
                    return_planet_positions=False,
                    interpolation='bilinear',
                )
            )

    # Apply temporal binning to the stack
    stack = prestack_array(array=stack, stacking_factor=binning_factor)

    return stack


def load_dataset(
    name_or_path: Union[str, Path],
    binning_factor: int = 1,
    frame_size: Optional[Tuple[int, int]] = None,
    remove_planets: bool = False,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    ObservingConditions,
    Dict[str, Union[str, float]],
]:
    """
    Convenience wrapper to load the stack, the parallactic angles, the
    PSF template, the observing conditions and the metadata all at once.

    Args:
        name_or_path: Name of a data set (e.g., "beta_pictoris__lp"),
            or Path to an HDF file that contains the data set.
        binning_factor: Number of frames that should be temporally
            binned ("pre-stacked") using a block-wise mean.
        frame_size: Target frame size to which the stack should be
            (spatially) cropped.
        remove_planets: If yes, negative fake planets are injected at
            the positions of the known planets to remove them from the
            stack. Useful for experiments with fake planets, for which
            a "clean" stack is required.

    Returns:
         A 5-tuple: `(stack, parang, psf_template, obs_con, metadata)`.
    """

    stack = load_stack(
        name_or_path=name_or_path,
        binning_factor=binning_factor,
        frame_size=frame_size,
        remove_planets=remove_planets,
    )
    parang = load_parang(
        name_or_path=name_or_path,
        binning_factor=binning_factor,
    )
    psf_template = load_psf_template(
        name_or_path=name_or_path,
    )
    observing_conditions = load_observing_conditions(
        name_or_path=name_or_path,
        binning_factor=binning_factor,
    )
    metadata = load_metadata(
        name_or_path=name_or_path,
    )

    return stack, parang, psf_template, observing_conditions, metadata


def get_field_rotation(parang: np.ndarray) -> float:
    """
    Compute the field rotation from a given array of parallactic angles.

    Args:
        parang: A 1D numpy array of shape `(n_frames,)` that contains
            the parallactic angle for each frame in degree.

    Returns:
        The field rotation in degree.
    """

    # Compute field rotation
    field_rotation = float(abs(parang[-1] - parang[0]))

    # If the value is physically sensible, we can return it
    if field_rotation < 180:
        return field_rotation

    # If the observation crosses the zenith, we sometimes get "jumps" in the
    # parallactic angle that break the computation of the field rotation. By
    # ensuring all angles are positive, we can try to fix this issue.
    new_parang = (parang + 360) % 360
    field_rotation = abs(new_parang[-1] - new_parang[0])
    if field_rotation < 180:
        return field_rotation

    # If the value is still not physically sensible, raise an error
    raise RuntimeError('Field rotation is greater than 180 degrees!')
