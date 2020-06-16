"""
Utility functions related to dealing with observing conditions.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
    key_map['m1_temperature'] = \
        dict(start_key='HIERARCH ESO TEL TH M1 TEMP',
             end_key='HIERARCH ESO TEL TH M1 TEMP')
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
    parameter: Union[str, Sequence[str]],
    long_description: bool = False,
) -> Union[Tuple[str, Optional[str]], List[Tuple[str, Optional[str]]]]:
    """
    Get the description and unit (as strings) for a given parameter,
    or a sequence of parameters.

    Args:
        parameter: A string (or a sequence of strings) containing the
            name(s) of the observing conditions parameter(s) whose
            description and unit we want to retrieve.
        long_description: If `True`, a longer version of the description
            is returned.

    Returns:
        A tuple `(description, unit)`, or a list of such tuples, which
        contains the description and unit of the target parameter as
        strings. For dimensionless parameters such as the relative air
        mass, `None` is returned as the unit.
    """

    # Define the look-up table for all descriptions and units
    descriptions_and_units: Dict[str, dict] = dict(
        air_mass=dict(
            short='Air mass',
            long='Air mass (relative to zenith)',
            unit=None,
        ),
        air_pressure=dict(
            short='Air pressure',
            long='Observatory ambient air pressure',
            unit='hPa',
        ),
        average_coherence_time=dict(
            short='Average coherence time',
            long='Average coherence time',
            unit='s',
        ),
        cos_wind_direction=dict(
            short='cos(wind direction)',
            long='Cosine of observatory ambient wind direction',
            unit=None,
        ),
        integrated_water_vapor=dict(
            short='Integrated Water Vapor',
            long='Integrated Water Vapor',
            unit='?',
        ),
        ir_sky_temperature=dict(
            short='IR sky temperature',
            long='Temperature of the IR sky',
            unit='?',
        ),
        m1_temperature=dict(
            short='M1 Temperature',
            long='Superficial temperature of mirror M1',
            unit='°C',
        ),
        observatory_temperature=dict(
            short='Observatory temperature',
            long='Observatory ambient temperature',
            unit='°C',
        ),
        relative_humidity=dict(
            short='Relative humidity',
            long='Observatory ambient relative humidity',
            unit='%',
        ),
        seeing=dict(
            short='Observatory seeing',
            long='Observatory seeing (before AO corrections)',
            unit='arcsec',
        ),
        sin_wind_direction=dict(
            short='sin(wind direction)',
            long='Sine of observatory ambient wind direction',
            unit=None,
        ),
        wind_direction=dict(
            short='Wind direction',
            long='Observatory ambient wind direction',
            unit='°',
        ),
        wind_speed=dict(
            short='Wind speed',
            long='Observatory ambient wind speed',
            unit='m/s',
        )
    )

    # Make sure that `parameter` is always a list, so that we can loop over it
    if isinstance(parameter, str):
        parameter = [parameter]

    # Initialize list of results
    results: List[Tuple[str, Optional[str]]] = list()

    # Define the key for accessing the right description type
    description_key = 'long' if long_description else 'short'

    # Loop over all requested parameters and resolve them
    for param in parameter:
        description: str = descriptions_and_units[param][description_key]
        unit: Optional[str] = descriptions_and_units[param]['unit']
        results.append((description, unit))

    # Return either a single tuple (if only one parameter was requested), or
    # a list of tuples (if multiple parameters where requested)
    if len(results) == 1:
        return results[0]
    return results


def load_observing_conditions(
    file_path: str,
    parameters: Optional[Iterable[str]] = 'all',
    transform_wind_direction: bool = True,
    coherence_time_in_ms: bool = True,
    as_dataframe: bool = False,
) -> Optional[Union[Dict[str, np.ndarray], pd.DataFrame]]:
    """
    Convenience wrapper for loading observing conditions from HDF files.

    Args:
        file_path: Path to the HDF file containing the observing
            conditions.
        parameters: An iterable of strings, containing the names of the
            parameters to be loaded. If "all" is given, all available
            parameters are loaded. If `None` is given, an empty dict or
            DataFrame is returned.
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
        conditions. May be empty, if parameter is None.
    """

    # Initialize dictionary to hold observing conditions
    observing_conditions: Dict[str, np.ndarray] = dict()

    # Load observing conditions from HDF file
    with h5py.File(file_path, 'r') as hdf_file:

        # If no list of parameters was given, use all available parameters
        if parameters == 'all':
            parameters = sorted(list(hdf_file.keys()))
        elif parameters is None:
            parameters = list()

        # Loop over parameter and load them from the HDF file
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
