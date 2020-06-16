"""
Utility functions related to dealing with observing conditions.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Dict, Iterable, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

def get_key_map(
    instrument: str = 'NACO',
) -> Dict[str, Dict[str, str]]:
    """
    Return a dictionary that maps the "intuitive" names of relevant
    observing condition parameters to the respective keys used in the
    headers of ESO/VLT FITS files.

    As the set of available parameters is instrument-specific (not all
    keys exist for all instruments), this function also takes the name
    of the instrument as an input.

    Args:
        instrument: A string (either "NACO" or "IRDIS") containing the
            name of the (sub)-instrument for which to get the key map.

    Returns:
        A dictionary mapping intuitive parameter names to the ones used
        in the header of a FITS file.
    """

    # Make sure we have received a valid value for the instrument
    if instrument not in ('NACO', 'IRDIS'):
        raise ValueError('Invalid value for "instrument"!')

    # Initialize the key map
    key_map = dict()

    # Add keys that exist for all instruments
    key_map['air_mass'] = \
        dict(start_key='HIERARCH ESO TEL AIRM START',
             end_key='HIERARCH ESO TEL AIRM END')
    key_map['air_pressure'] = \
        dict(start_key='HIERARCH ESO TEL AMBI PRES START',
             end_key='HIERARCH ESO TEL AMBI PRES END')
    key_map['average_coherence_time'] = \
        dict(start_key='HIERARCH ESO TEL AMBI TAU0',
             end_key='HIERARCH ESO TEL AMBI TAU0')
    key_map['observatory_temperature'] = \
        dict(start_key='HIERARCH ESO TEL AMBI TEMP',
             end_key='HIERARCH ESO TEL AMBI TEMP')
    key_map['relative_humidity'] = \
        dict(start_key='HIERARCH ESO TEL AMBI RHUM',
             end_key='HIERARCH ESO TEL AMBI RHUM')
    key_map['seeing'] = \
        dict(start_key='HIERARCH ESO TEL AMBI FWHM START',
             end_key='HIERARCH ESO TEL AMBI FWHM END')
    key_map['wind_direction'] = \
        dict(start_key='HIERARCH ESO TEL AMBI WINDDIR',
             end_key='HIERARCH ESO TEL AMBI WINDDIR')
    key_map['wind_speed'] = \
        dict(start_key='HIERARCH ESO TEL AMBI WINDSP',
             end_key='HIERARCH ESO TEL AMBI WINDSP')

    # Add keys that only exist for NACO
    if instrument == 'NACO':
        key_map['integrated_water_vapor'] = \
            dict(start_key='HIERARCH ESO TEL AMBI IWV START',
                 end_key='HIERARCH ESO TEL AMBI IWV END')
        key_map['ir_sky_temperature'] = \
            dict(start_key='HIERARCH ESO TEL AMBI IRSKY TEMP',
                 end_key='HIERARCH ESO TEL AMBI IRSKY TEMP')

    # Add keys that only exist for IRDIS
    if instrument == 'IRDIS':
        pass

    # Make sure the dict is sorted. This only works for Python 3.7 and up!
    key_map = {k: key_map[k] for k in sorted(key_map)}

    return key_map


def get_description_and_unit(
    parameter: str,
) -> Tuple[str, Optional[str]]:
    """
    For a given observing conditions parameter (as a string), return the
    corresponding description and unit (as strings).

    Returns:
        A tuple (description, unit) containing the description and unit
        of the target parameter as strings. For dimensionless parameters
        such as the relative air mass, `None` is returned as the unit.
    """

    if parameter == 'air_mass':
        description = 'Air mass (relative to zenith)'
        unit = None
    elif parameter == 'seeing':
        description = 'Observatory seeing (before AO corrections)'
        unit = 'arcsec'
    elif parameter == 'ir_sky_temperature':
        description = 'Temperature of the IR sky'
        unit = '?'
    elif parameter == 'integrated_water_vapor':
        description = 'Integrated Water Vapor'
        unit = '?'
    elif parameter == 'air_pressure':
        description = 'Observatory ambient air pressure'
        unit = 'hPa'
    elif parameter == 'relative_humidity':
        description = 'Observatory ambient relative humidity'
        unit = '%'
    elif parameter == 'average_coherence_time':
        description = 'Average coherence time'
        unit = 's'
    elif parameter == 'observatory_temperature':
        description = 'Observatory ambient temperature'
        unit = 'degree Celsius'
    elif parameter == 'wind_direction':
        description = "Observatory ambient wind direction"
        unit = 'degree'
    elif parameter == 'cos_wind_direction':
        description = "Cosine of observatory ambient wind direction"
        unit = None
    elif parameter == 'sin_wind_direction':
        description = "Sine of observatory ambient wind direction"
        unit = None
    elif parameter == 'wind_speed':
        description = "Observatory ambient wind speed"
        unit = 'm/s'
    else:
        raise ValueError(f'Unknown parameter name: "{parameter}"!')

    return description, unit


def load_observing_conditions(
    file_path: str,
    parameters: Optional[Iterable[str]] = None,
    transform_wind_direction: bool = True,
    coherence_time_in_ms: bool = True,
    as_dataframe: bool = False,
) -> Union[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Convenience wrapper for loading observing conditions from HDF files.

    Args:
        file_path: Path to the HDF file containing the observing
            conditions.
        parameters: An iterable of strings, containing the names of the
            parameters to be loaded. If `None` is given, all available
            parameters are loaded.
        transform_wind_direction: If True, do not return the values for
            `wind_direction` directly, but instead return the cosine and
            sine (useful if you want to use the wind direction as a
            predictor in an ML model, because, for example, 1 degree
            and 359 degree should be "close").
        coherence_time_in_ms: Whether or not to convert the average
            coherence time (tau_0) from seconds to milliseconds.
        as_dataframe: If True, the data is returned as a pandas
            DataFrame instead of a simple dictionary.

    Returns:
        Either a dictionary or a pandas DataFrame containing the values
        of the requested `parameters` from the given file of observing
        conditions.
    """

    # If no list of parameters was given, use all available parameters
    if parameters is None:
        parameters = list(get_key_map().keys())

    # Initialize dictionary to hold observing conditions
    observing_conditions: Dict[str, np.ndarray] = dict()

    # Load observing conditions from HDF file
    with h5py.File(file_path, 'r') as hdf_file:
        for key in parameters:
            observing_conditions[key] = np.array(hdf_file[key])

    # Return sine and cosine of wind direction (instead of degrees)
    if 'wind_direction' in parameters and transform_wind_direction:
        observing_conditions['cos_wind_direction'] = \
            np.cos(np.deg2rad(observing_conditions['wind_direction']))
        observing_conditions['sin_wind_direction'] = \
            np.sin(np.deg2rad(observing_conditions['wind_direction']))
        del observing_conditions['wind_direction']

    # Convert average coherence time from seconds to milliseconds
    if 'average_coherence_time' in parameters and coherence_time_in_ms:
        observing_conditions['average_coherence_time'] *= 1000

    # Return observing conditions either as a data frame or as a dictionary
    if as_dataframe:
        return pd.DataFrame(observing_conditions)
    return observing_conditions
