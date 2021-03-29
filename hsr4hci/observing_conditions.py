"""
Utility functions related to dealing with observing conditions.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from datetime import datetime, timezone
from typing import Dict, List, Tuple, Union

import io
import math

from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import convolve

import numpy as np
import pandas as pd
import requests

from hsr4hci.time_conversion import (
    timestamp_to_date_string,
    timestamp_to_datetime,
)


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class ObservingConditions:
    """
    This class provides a wrapper around different representations of
    the observing conditions (dictionary, numpy array and pandas data
    frame), and provides the option to select a subset of them.

    Args:
        observing_conditions: The observing conditions in the form of
             a dictionary where each key maps onto a 1D numpy array
             that contains, for example, the wind speed.
    """

    def __init__(self, observing_conditions: Dict[str, np.ndarray]):
        self.observing_conditions = observing_conditions

    def _verify_selected_keys(
        self, selected_keys: Union[List[str], str, None]
    ) -> None:
        """
        Make sure that the `selected_keys` constitute a valid subset
        of the available observing conditions.
        """

        # Check if `selected_keys` contains a valid subset selection
        if (not selected_keys) or (selected_keys is None):
            return
        if selected_keys == 'all':
            return
        if isinstance(selected_keys, list) and all(
            _ in self.available_keys for _ in selected_keys
        ):
            return

        # Otherwise, raise an error
        raise ValueError(
            f'{selected_keys} is not a valid value for selected_keys!'
        )

    @property
    def available_keys(self) -> List[str]:
        """
        Return a sorted list of the available observing conditions.

        Returns:
            A a sorted list of the available observing conditions.
        """
        return sorted(list(self.observing_conditions.keys()))

    @property
    def n_frames(self) -> int:
        """
        Get the number of frames from the observing conditions.

        Returns:
            An integer containing the number of frames.
        """
        return len(next(iter(self.observing_conditions.values())))

    def as_dict(
        self,
        selected_keys: Union[List[str], str, None] = 'all',
    ) -> Dict[str, np.ndarray]:
        """
        Return the subset of observing conditions selected by
        `selected_keys` as a dictionary.

        Args:
            selected_keys: A valid specification of a subset of
                observing conditions. Either None (to not select any
                observing) conditions, or a list of keys, or "all" (to
                select all available observing conditions).

        Returns:
            A dictionary containing the selected observing conditions.
        """

        # Make sure are selecting a valid subset
        self._verify_selected_keys(selected_keys)

        # If we do not select any keys, return an empty dictionary
        if (not selected_keys) or (selected_keys is None):
            return {}

        # Resolve 'all' into a list of all available observing conditions
        if selected_keys == 'all':
            selected_keys = sorted(list(self.observing_conditions.keys()))

        # Otherwise, return the selected subset of self.observing_conditions
        return {
            k: v
            for k, v in self.observing_conditions.items()
            if k in selected_keys
        }

    def as_array(
        self,
        selected_keys: Union[List[str], str, None] = 'all',
    ) -> np.ndarray:
        """
        Return the subset of observing conditions selected by
        `selected_keys` as a numpy array.

        Args:
            selected_keys: A valid specification of a subset of
                observing conditions. Either None (to not select any
                observing) conditions, or a list of keys, or "all" (to
                select all available observing conditions).

        Returns:
            A 2D numpy array of shape `(n_frames, n_obscon)` containing
            the selected observing conditions.
        """

        # Make sure are selecting a valid subset
        self._verify_selected_keys(selected_keys)

        # Get the selected observing conditions as a dictionary
        observing_conditions = self.as_dict(selected_keys)

        # If no keys were selected (i.e., selected_keys was either None or an
        # empty list), we return an empty 2D array of shape `(n_frames, 0)`.
        # In this form, it  can still be concatenated to the predictors of an
        # HSR model, so we do not need to take special care of it later.
        if not observing_conditions:
            return np.empty((self.n_frames, 0))

        # Otherwise, we can convert the dictionary to a 2D numpy array
        return np.hstack(
            [_.reshape(-1, 1) for _ in observing_conditions.values()]
        )

    def as_dataframe(
        self,
        selected_keys: Union[List[str], str, None] = 'all',
    ) -> pd.DataFrame:
        """
        Return the subset of observing conditions selected by
        `selected_keys` as a pandas data frame.

        Args:
            selected_keys: A valid specification of a subset of
                observing conditions. Either None (to not select any
                observing) conditions, or a list of keys, or "all" (to
                select all available observing conditions).

        Returns:
            A pandas data frame containing the selected observing
            conditions.
        """

        # Make sure are selecting a valid subset
        self._verify_selected_keys(selected_keys)

        # Get the selected observing conditions as a dictionary
        observing_conditions = self.as_dict(selected_keys)

        # Convert the observing conditions to a data frame and return them
        return pd.DataFrame(observing_conditions)


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def query_archive(
    start_date: str,
    end_date: str,
    archive: str,
    parameter_key: str,
) -> pd.DataFrame:
    """
    Send a request to one of ESO's ambient condition query forms [1] to
    retrieve the values of a particular observing condition.

    [1]: http://archive.eso.org/cms/eso-data/ambient-conditions/paranal-ambient-query-forms.html

    Args:
        start_date: The start datetime (in UTC) as a string in ISO 8061
            format. Example: "2012-12-20T20:00:00.0000".
        end_date: The end datetime (in UTC) as a string in ISO 8061
            format. Example: "2012-12-21T10:00:00.0000".
        archive: The name of the archive to which to send the query.
            Currently, the following archives are supported:
                - "meteo"
                - "dimm_old"
                - "dimm_new"
                - "mass"
                - "lhatpro"
                - "lhatpro_irt"
            See [1] for more information about these archives.
        parameter_key: The key under which a parameter is available from
            the respective archive. These keys can be reverse-engineered
            from the source code of the respective archive. For example,
            "press" will get you the air pressure, or more precisely,
            the "temporal (1 minute) mean of observatory site ambient
            barometric air pressure measured at ground during
            measurement period [hPa]."
            For the default parameters, `resolve_parameter_name()` will
            resolve "intuitive" parameter names (like "air_pressure") to
            the correct, archive-specific keys.

    Returns:
        A data frame with the datetime and timestamp, the integration
        time (in seconds) and the value of the requested parameter
        averaged over the integration time.
    """

    # Define the payload for the query
    data = {
        'wdbo': 'csv/download',
        'max_rows_returned': 10_000_000,
        'start_date': f'{start_date}..{end_date}',
        'tab_integration': 'on',
        f'tab_{parameter_key}': 'on',
        'order': 'start_date',
    }

    # Define the URL for query based on the archive
    if archive == 'meteo':
        url = 'https://archive.eso.org/wdb/wdb/asm/meteo_paranal/query'
    elif archive == 'dimm_old':
        url = 'https://archive.eso.org/wdb/wdb/asm/historical_ambient_paranal/query'
    elif archive == 'dimm_new':
        url = 'https://archive.eso.org/wdb/wdb/asm/dimm_paranal/query'
    elif archive == 'mass':
        url = 'https://archive.eso.org/wdb/wdb/asm/mass_paranal/query'
    elif archive == 'lhatpro':
        url = 'https://archive.eso.org/wdb/wdb/asm/lhatpro_paranal/query'
    elif archive == 'lhatpro_irt':
        url = 'https://archive.eso.org/wdb/wdb/asm/lhatpro_irt_paranal/query'
    else:
        raise ValueError(f'Invalid archive: "{archive}"!')

    # Send a POST request to the "Meteorology Query Form"; raise error
    # if the return code is 4XX or 5XX.
    response = requests.post(url=url, data=data)
    response.raise_for_status()

    # Parse response (which is a CSV) to a pandas data frame
    df = pd.read_csv(
        filepath_or_buffer=io.StringIO(response.content.decode('utf-8')),
        names=['datetime', 'integration_time', parameter_key],
        header=0,
        parse_dates=['datetime'],
        date_parser=pd.to_datetime,
        comment='#',
    )

    # Add a column to the data frame for the timestamp
    df['timestamp'] = df.datetime.values.astype(np.int64) // 10 ** 9
    df.sort_values(by='timestamp', inplace=True)

    return df


def interpolate_observing_conditions(
    timestamps: np.ndarray,
    df: pd.DataFrame,
    parameter_key: str,
    method: str = 'spline',
) -> np.ndarray:
    """
    Take the values of the observing conditions in the data frame `df`
    and apply a special interpolation algorithm so that we can apply
    them at the given frame `timestamps`.

    Args:
        timestamps: A 1D numpy array of floats, containing the UTC
            timestamps of the frames in the stack.
        df: A data frame containing the result from querying one of
            the ESO archives (e.g., `query_meteo`).
        parameter_key: The key under which a parameter is available from
            the respective archive. (See also `query_archive()`).
        method: Which interpolation method to use. The options are
            "spline", for an interpolation based on Cubic splines (see
            https://stats.stackexchange.com/a/511394 for the idea), or
            "rymes-myers" to use the algorithm described in Rymes and
            Myers (2001); see below for full reference.
            The two methods generally give very similar results, but
            the spline interpolation is *much* faster, and due to its
            simplicity, it is also less likely to contain a bug :)

    Returns:
        A 1D numpy array (of length `n_frames`) which contains an
        interpolated value of the target parameter for every frame.
    """

    # -------------------------------------------------------------------------
    # Spline interpolation
    # -------------------------------------------------------------------------

    if method == 'spline':

        # Define shortcuts
        avg = df[parameter_key].values
        x = df['timestamp'].values

        # Remove NaNs, because they break the spline interpolation
        nan_idx = np.isnan(avg)
        avg = avg[~nan_idx]
        x = x[~nan_idx]

        # Compute y, which is essentially the cumulative sum of the parameter
        y = np.zeros(len(x))
        for i in range(1, len(x)):
            y[i] = y[i - 1] + avg[i - 1] * (x[i] - x[i - 1])

        # Set up an interpolation using splines. We use the first derivative
        # here, because the cumulative sum above is basically an integral.
        interpolator = CubicSpline(x, y).derivative(1)

        # Evaluate the interpolator at the time of each frame
        return np.asarray(interpolator(timestamps))

    # -------------------------------------------------------------------------
    # Rymes-Myers interpolation (more complicated and much slower!)
    # -------------------------------------------------------------------------

    if method == 'rymes-myers':

        # Get the start and end date for the query (as strings). We need an
        # offset of the observation duration both before and after the first
        # and last frame for interpolation purposes (see below).
        duration = int(math.ceil(max(timestamps) - min(timestamps)))

        # Find the intervals limits: by default, the ESO archive only returns
        # the average over a 60 second interval for a parameter. Here, we
        # compute the start and end timestamp for each of these intervals.
        interval_limits = np.column_stack(
            [
                (df['timestamp'] - df['integration_time']).values[1:],
                df['timestamp'].values[1:],
            ]
        ).astype(int)

        # Create an "upsampled" version of the time series of the parameter of
        # interest: the upsampled version has a temporal resolution of 1 second
        # all points within one interval (see above) have the same value.
        upsampler = interp1d(
            x=df['timestamp'].values,
            y=df[parameter_key].values,
            kind='previous',
        )
        upsampled_timestamps = np.arange(
            min(df['timestamp']), max(df['timestamp']) + 1
        )
        upsampled_values = upsampler(upsampled_timestamps)

        # Determine the offset that is required to convert between timestamps
        # and array indices
        offset = int(upsampled_timestamps[0])

        # In the following, we apply the mean-preserving interpolation
        # algorithm from:
        #
        #   Rymes, M. D., & Myers, D. R. (2001). "Mean preserving algorithm
        #       for smoothly interpolating averaged data." Solar Energy,
        #       71(4), p. 225–231. DOI:10.1016/s0038-092x(01)00052-4.
        #
        # Essentially, what this algorithm does is to iteratively take the
        # rolling average of the upsampled date, and then at each iteration
        # correct the mean in every interval to the original value.
        # This procedure is not particularly sophisticated and will probably
        # not give us the true parameter values, but the result is reasonably
        # smooth and if we do take the interval-wise average, we get back the
        # original time series from the ESO archive.

        # Prepare the kernel for the rolling average, and the array that we
        # will iteratively convolve with the kernel
        kernel = np.ones(3) / 3
        interpolated = np.copy(upsampled_values)

        # We have to repeat the procedure N times, where N is the number of
        # time steps in the target time series
        for i in range(int(duration)):

            # Compute the rolling average
            interpolated = convolve(interpolated, kernel, mode='same')

            # Loop over the intervals and adjust their mean. Note that the
            # original publication seems to have a sign error in the last
            # step (they subtract C_k where they should add it).
            for a, b in interval_limits:
                avg_k = float(upsampled_values[int((a + b) / 2) - offset])
                mean_k = float(
                    np.mean(interpolated[(a - offset) : (b - offset)])
                )
                interpolated[(a - offset) : (b - offset)] += avg_k - mean_k

        # Set up another interpolator that allows us the compute the value of
        # the target parameter at an arbitrary time between the start and end
        # of the observation.
        # Note that this interpolator does, in principle, return values for
        # the interval [start - duration, end + duration]. However, the ranges
        # [start - duration, start] and [end, end + duration] must NOT be used
        # as they contain unphysical values: basically, because of the use of
        # `mode='same'`, these values get convolved with a lot of zeros from
        # the (implicit) padding of the np.convolve() command.
        interpolator = interp1d(upsampled_timestamps, interpolated)

        # Evaluate the interpolator at the time of each frame
        return np.asarray(interpolator(timestamps))

    # -------------------------------------------------------------------------
    # Value error for invalid methods
    # -------------------------------------------------------------------------

    raise ValueError(f'Invalid method: "{method}"!')


def get_observing_conditions(
    parameter_name: str,
    timestamps: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    This is a convenience wrapper to query the ESO ambient condition
    archives for a given parameter, interpolate the results, and
    evaluate them at the requested `timestamps`.

    Args:
        parameter_name: Name of the parameter to retrieve from the
            archive. This needs to be resolvable into a parameter key
            and archive by `resolve_parameter_name()`.
        timestamps: A 1D numpy array of floats, containing the UTC
            timestamps of the frames in the stack.

    Returns:
        A 1D numpy array (of length `n_frames`) which contains an
        interpolated value of the target parameter for every frame.
    """

    # Get the start and end date for the query (as strings). We need an
    # offset of the observation duration both before and after the first
    # and last frame for interpolation purposes (see below).
    duration = int(math.ceil(max(timestamps) - min(timestamps)))
    start_timestamp = math.floor(min(timestamps) - duration - 60)
    end_timestamp = math.ceil(max(timestamps) + duration + 60)

    # Resolve the parameter name
    archive, key, _ = resolve_parameter_name(
        parameter_name=parameter_name,
        obs_date=timestamp_to_datetime(start_timestamp),
    )

    # Query the archive to get data frame with parameter of interest
    df = query_archive(
        start_date=timestamp_to_date_string(start_timestamp),
        end_date=timestamp_to_date_string(end_timestamp),
        parameter_key=key,
        archive=archive,
    )

    # Convert the data frame to a dictionary (for storing them in the HDF file)
    query_results = dict(
        timestamp=df['timestamp'].values,
        integration_time=df['integration_time'].values,
        parameter=df[key].values,
    )

    # Interpolate the observing conditions for each frame
    interpolated = interpolate_observing_conditions(timestamps, df, key)

    return interpolated, query_results


def resolve_parameter_name(
    parameter_name: str,
    obs_date: datetime,
) -> Tuple[str, str, str]:
    """
    Resolves a `parameter_name` into a dictionary that contains
    information about which archive the parameter can be obtained
    from, using which parameter_key.

    Args:
        parameter_name: Name of a parameter (e.g., 'air_pressure').
        obs_date: Date at which the data set was observed.

    Returns:
        A tuple `archive`, `parameter_key`, `description` which tells
        us from which ESO ambient server we can retrieve the parameter
        and which key we have to use for the query.
    """

    # -------------------------------------------------------------------------
    # Preliminaries
    # -------------------------------------------------------------------------

    # For non-meteorological parameters, the archive depends on the observation
    # date, as the Astronomical Site Monitoring (ASM) was updated in April 2016
    if obs_date > datetime(2016, 4, 4, 12, 0, 0, 0, timezone.utc):
        archive_version = 'new'
    else:
        archive_version = 'old'

    # Initialize default values
    archive = ''
    parameter_key = ''
    description = ''

    # -------------------------------------------------------------------------
    # Resolve the parameter name
    # -------------------------------------------------------------------------

    # Air pressure
    if parameter_name == 'air_pressure':
        archive = 'meteo'
        parameter_key = 'press'
        description = (
            'Temporal (1 minute) mean of observatory site ambient barometric '
            'air pressure measured at ground during measurement period [hPa].'
        )

    # Coherence time (tau_0)
    elif parameter_name == 'coherence_time':
        if archive_version == 'old':
            archive = 'dimm_old'
            parameter_key = 'tau'
            description = 'Coherence time [s].'
        if archive_version == 'new':
            archive = 'mass'
            parameter_key = 'tau'
            description = (
                'Coherence time (weights method) from MASS stand-alone [s].'
            )

    # Isoplanatic angle (theta_0)
    elif parameter_name == 'isoplanatic_angle':
        if archive_version == 'old':
            archive = 'dimm_old'
            parameter_key = 'tet'
            description = 'Isoplanatic angle [arcsec].'
        if archive_version == 'new':
            archive = 'mass'
            parameter_key = 'tet'
            description = (
                'Isoplanatic angle from MASS-DIMM integrated profile [J1:J6] '
                '[arcsec].'
            )

    # Observatory temperature
    elif parameter_name == 'observatory_temperature':
        archive = 'meteo'
        parameter_key = 'temp2'
        description = (
            'Temporal (1 minute) mean of site ambient temperature measured '
            'at 2m [deg Celsius].'
        )

    # Relative humidity
    elif parameter_name == 'relative_humidity':
        archive = 'meteo'
        parameter_key = 'rhum1'
        description = (
            'Temporal (1 minute) mean of observatory site ambient relative '
            'humidity measured at sensor position 30m above ground during '
            'measurement period [%].'
        )

    # Seeing
    elif parameter_name == 'seeing':
        if archive_version == 'old':
            archive = 'dimm_old'
            parameter_key = 'fwhm'
            description = (
                'Reference observatory site seeing measured by the ASM-DIMM '
                'telescope, Full Width Half Maximum at 500nm [arcsec].'
            )
        if archive_version == 'new':
            archive = 'dimm_new'
            parameter_key = 'fwhm'
            description = (
                'The total seeing calculated with DIMM telescope [arcsec]. '
                'The value is calculated using the following formula: '
                'FWHM = 2E(+7) Cn2**(0.6).'
            )

    # Wind speed (U component)
    elif parameter_name == 'wind_speed_u':
        archive = 'meteo'
        parameter_key = 'wind_speedu'
        description = (
            'Temporal (1 minute) mean of observatory site ambient wind '
            'speed U vector component, where U is horizontal and points '
            'to 330 degree measured at sensor position 20m during '
            'measurement period [m/s].'
        )

    # Wind speed (V component)
    elif parameter_name == 'wind_speed_v':
        archive = 'meteo'
        parameter_key = 'wind_speedv'
        description = (
            'Temporal (1 minute) mean of observatory site ambient wind '
            'speed V vector component, where V is horizontal and points '
            'to 240 degree measured at sensor position 20m during '
            'measurement period [m/s].'
        )

    # Wind speed (W component)
    elif parameter_name == 'wind_speed_w':
        archive = 'meteo'
        parameter_key = 'wind_speedw'
        description = (
            'Temporal (1 minute) mean of observatory site ambient wind '
            'speed W vector component, where W is vertically pointing '
            'upwards, measured at sensor position 20m during measurement '
            'period [m/s]. '
        )

    # For all other parameter names, raise an error
    else:
        raise ValueError(f'Invalid parameter_name: "{parameter_name}"!')

    return archive, parameter_key, description
