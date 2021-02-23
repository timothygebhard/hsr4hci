"""
Utilities for extending the abilities of PynPoint.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from datetime import datetime, timedelta, timezone
from warnings import warn

import time

from astropy.time import Time

import numpy as np

from pynpoint.core.processing import ProcessingModule
from pynpoint.util.module import progress


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class TimestampModule(ProcessingModule):
    """
    Module for calculating the UNIX timestamp of each frame (because
    HDF5 does not have a dtype for datetimes and this is the easiest
    way of keeping track of the exact date and time of each frame).

    NOTE: This module currently only supports NACO data.

    Args:
        name_in: Unique name of the module instance.
        data_tag: Tag of the database entry for which the timestamp
            values are written as attributes.
    """

    def __init__(self, name_in: str, data_tag: str) -> None:

        super().__init__(name_in)

        self.m_data_in_port = self.add_input_port(data_tag)
        self.m_data_out_port = self.add_output_port(data_tag)

    @staticmethod
    def _attribute_check(ndit: np.ndarray, steps: np.ndarray) -> None:

        if not np.all(ndit == steps):
            warn('There is a mismatch between the NDIT and NFRAMES values!')

    def run(self) -> None:
        """
        Run method of the module.

        Calculates the datetime (in UTC) for each frame and converts
        it to a UNIX timestamp (as a float). The values are written
        as TIMESTAMP_UTC attributes to the *data_tag*.
        """

        # Load cube sizes (and check consistency)
        steps = self.m_data_in_port.get_attribute('NFRAMES')
        ndit = self.m_data_in_port.get_attribute('NDIT')
        self._attribute_check(ndit, steps)

        # Load detector integration time (in seconds)
        dit = self.m_data_in_port.get_attribute('DIT')

        # Load start times (i.e., times of the first frame in each cube)
        # For NACO observations, the 'DATE' field is, by default, mapped to
        # the 'DATE-OBS' field in the FITS header, which contains the date
        # at which the exposure was started. (This behaviour is configured
        # in the PynPoint_config.ini file.)
        # The value string for the date uses the restricted ISO 8601 format,
        # that is, 'YYYY-MM-DDThh:mm:ss.sss'. Unless there is an additional
        # 'TIMESYS' field in the FITS header specifying specifying another
        # time system, the time zone can be assumed to be UTC.
        # Check out the "ESO Data Interface Control Document" for details.
        cube_start_times = self.m_data_in_port.get_attribute('DATE')

        # Keep track of the timestamps of all frames
        all_timestamps = []

        # Start the stopwatch for the loop over the cubes
        loop_start = time.time()

        # Calculate datetimes for for each cube
        for i, n_frames_in_cube in enumerate(steps):

            # Show progress bar
            progress(
                current=i,
                total=len(steps),
                message='Calculating datetimes...',
                start_time=loop_start,
            )

            # Get the start time of the current cube: use astropy to parse the
            # FITS data format, and then convert to a regular datetime object
            cube_start = Time(
                val=cube_start_times[i].decode('utf-8'),
                format='fits',
                scale='utc',
            ).to_datetime(timezone=timezone.utc)

            # Compute timestamp for each frame in the cube
            for j in range(n_frames_in_cube):
                frame_datetime = cube_start + timedelta(seconds=(j * dit))
                frame_timestamp = datetime.timestamp(frame_datetime)
                all_timestamps.append(frame_timestamp)

        # Write the final result (i.e., the timestamps for all frames in all
        # cubes) to the PynPoint database
        self.m_data_out_port.add_attribute(
            'TIMESTAMP_UTC', np.array(all_timestamps), static=False
        )
