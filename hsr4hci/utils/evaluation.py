"""
Utility functions for performance evaluation (e.g., computing the SNR).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Dict, Optional, Tuple
from warnings import warn

from contexttimer.timeout import timeout
from pynpoint.util.analysis import false_alarm
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize

import numpy as np


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class TimeoutException(Exception):
    """
    A custom exception for functions that exceed their maximum runtime.
    """


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def timeout_handler(*_, **__):
    """
    This function only serves as a callback for the @timeout decorator
    to raise a TimeoutException.
    """
    raise TimeoutException()


def compute_figures_of_merit(frame: np.ndarray,
                             position: Tuple[float, float],
                             aperture_size: float,
                             ignore_neighbors: bool = True,
                             target: Optional[str] = None,
                             method: Optional[str] = 'Nelder-Mead',
                             max_distance: Optional[float] = 1.0,
                             time_limit: int = 30) -> Dict[str, Any]:
    """
    Compute the figures of merit for a given residual frame, namely
    the signal, the noise level, the signal-to-noise ratio (SNR),
    and the false positive fraction (FPF).

    This method relies on the definitions from:
        Mawet et al. (2014): "Fundamental limitations of high contrast
        imaging set by small sample statistics", arXiv:1407.2247

    In practice, we are using the implementation provided by PynPoint
    which we augment with some additional functionality.

    Args:
        frame: Residual frame for which to compute the figures of merit.
        position: Position for which to compute the figures of merit.
            If `optimize` is not `None`, this position is only the
            starting point for the optimization.
        aperture_size: Size (radius in pixels) of the apertures used for
            computing the noise estimate.
        ignore_neighbors: Whether or not the ignore the two closest
            apertures to the `position` (which, at least in the case of
            PCA-based PSF subtraction, often contain the characteristic
            "self-subtraction wings"). Default is True.
        target: Which figure of merit should be optimized by varying
            the position slightly. Choices are "signal", "noise_level",
            "snr" and "fpf". If None is given, no optimization is
            performed and the figures of merit are simply computed at
            the given position (this is the default).
        method: Which method to use for this optimization (as a string).
            Must be a method supported by scipy.optimize.minimize(); see
            the scipy docs for more details. Default is "Nelder-Mead".
        max_distance: When using an optimizer, this parameter controls
            the maximum (Euclidean) distance of the optimal position
            from the initial position (in pixels).
            This is particularly useful for faint sources, where we do
            not want the optimizer to just "wander off" too far from the
            initial position. If set to None, there is no such maximum
            distance. Default is 1 pixel.
        time_limit: The maximum runtime of the function in seconds. This
            is particularly useful when using an optimizer which may get
            stuck. Default value is 30 seconds.

    Returns:
        A dictionary containing the figures of merit using the
        following keys: {"signal",  "noise_level", "snr", "fpf",
        "old_position", "new_position", "message", "success"}.
    """

    # -------------------------------------------------------------------------
    # Define dummy function and wrap it with a @timeout decorator
    # -------------------------------------------------------------------------

    # We would like the time limit for this function to be definable as an
    # argument. Since the time limit is enforced using the @timeout decorator,
    # we therefore need to define a dummy function which encapsulates all the
    # real functionality, and which we can then decorate with the corresponding
    # timeout decorator (which takes the desired time limit as an argument).

    @timeout(limit=int(time_limit), handler=timeout_handler)
    def _compute_figures_of_merit(frame: np.ndarray,
                                  position: Tuple[float, float],
                                  aperture_size: float,
                                  ignore_neighbors: bool = True,
                                  target: Optional[str] = None,
                                  method: Optional[str] = 'Nelder-Mead',
                                  max_distance: Optional[float] = 1.0) -> dict:
        """
        This is a dummy function which is needed to limit the runtime.
        """

        # ---------------------------------------------------------------------
        # Find the position at which we compute the figures of merit
        # ---------------------------------------------------------------------

        # Check if there is a conflict between the target and the method
        # variable: If we have selected an optimization target, but no
        # optimizer, we revert to the default and raise a warning.
        if target is not None and method is None:
            method = 'Nelder-Mean'
            warn('Conflicting options for optimize and method: '
                 f'optimize = {target}, but method = {method}. '
                 'Defaulted to use method = "Nelder-Mead" instead.')

        # If we do not optimize anything, we simply keep the initial position
        # and use default value for the "optimization" success and message
        if target is None:
            x, y = position
            message = 'Figure of merit computation terminated successfully.'
            success = True

        # Otherwise, we first need to find the "optimal" position
        elif target in ('signal', 'noise_level', 'snr', 'fpf'):

            # Define another dummy function to get the FOM which we want to
            # optimize by adjusting the position. This is essentially just a
            # partial function application where we fix all arguments of the
            # false_alarm() function, except for the position.
            def _get_fom(pos):

                # If the current position (`pos`) is too far from the initial
                # position (`position`), we return default values which
                # indicate the the current position not an admissible result.
                distance = euclidean(position, pos)
                if (max_distance is not None) and (distance > max_distance):
                    signal, noise_level, snr, fpf = 0, np.inf, 0, np.inf

                # Otherwise, we can compute the real figures of merit at `pos`
                else:

                    # Evaluate the figures of merit at the given position
                    try:
                        signal, noise_level, snr, fpf = \
                            false_alarm(image=frame,
                                        x_pos=pos[0],
                                        y_pos=pos[1],
                                        size=aperture_size,
                                        ignore=ignore_neighbors)

                    # In case of an error, set some defaults which indicate the
                    # current position is not the optimum
                    except ValueError:
                        signal, noise_level, snr, fpf = 0, np.inf, 0, np.inf

                # Depending on the quantity, we either need to minimize x or -x
                if target == 'signal':
                    return -1 * signal
                if target == 'noise_level':
                    return noise_level
                if target == 'snr':
                    return -1 * snr
                if target == 'fpf':
                    return fpf

            # Actually run the optimization to find the optimal position
            optimum = minimize(fun=_get_fom,
                               x0=np.array(position),
                               method=method)
            x, y = np.round(optimum.x, 2)
            message = optimum.message
            success = optimum.success

        # Raise an error if we receive an invalid value for optimize
        else:
            raise ValueError('optimize must be one of the following: '
                             '[None, "signal", "noise_level", "snr", "fpf"]!')

        # ---------------------------------------------------------------------
        # Actually compute and return the figures of merit
        # ---------------------------------------------------------------------

        # Compute signal, noise level, SNR and FPF at the desired position
        signal, noise_level, snr, fpf = false_alarm(image=frame,
                                                    x_pos=x,
                                                    y_pos=y,
                                                    size=aperture_size,
                                                    ignore=ignore_neighbors)

        # Return everything, including the (potentially optimized) position
        return dict(signal=signal,
                    noise_level=noise_level,
                    snr=snr,
                    fpf=fpf,
                    old_position=position,
                    new_position=(x, y),
                    message=message,
                    success=success)

    # -------------------------------------------------------------------------
    # Run dummy function to compute figures of merit
    # -------------------------------------------------------------------------

    # Finally, we simply call the dummy function with the arguments passed to
    # this function. It it finished successfully, we simply return its output:
    try:

        return _compute_figures_of_merit(frame=frame,
                                         position=position,
                                         aperture_size=aperture_size,
                                         ignore_neighbors=ignore_neighbors,
                                         target=target,
                                         method=method,
                                         max_distance=max_distance)

    # If the computation fails with a timeout, we return default values:
    except TimeoutException:

        return dict(signal=None,
                    noise_level=None,
                    snr=None,
                    fpf=None,
                    old_position=position,
                    new_position=None,
                    message='Figure of merit computation timed out!',
                    success=False)
