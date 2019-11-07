"""
Methods for performance evaluation (e.g., computing the SNR).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Optional, Tuple

from pynpoint.util.analysis import false_alarm
from scipy.optimize import minimize

import numpy as np


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def compute_figures_of_merit(frame: np.ndarray,
                             position: Tuple[float, float],
                             aperture_size: float,
                             ignore_neighbors: bool,
                             optimize: Optional[str]) -> dict:
    """
    Compute the figures of merit for a given residual frame, namely
    the signal, the noise level, the signal-to-noise ratio (SNR),
    and the false positive fraction (FPF).

    This method relies on the definitions from:
        Mawet et al. (2014): "Fundamental limitations of high contrast
        imaging set by small sample statistics", arXiv:1407.2247

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
            "self-subtraction wings").
        optimize: Which figure of merit should be optimized by varying
            the position slightly. Choices are "signal", "noise_level",
            "snr" and "fpf". If None is given, no optimization is
            performed and the figures of merit are simply computed at
            the given position.

    Returns:
        A dictionary containing the figures of merit using the
        following keys: {"signal",  "noise_level", "snr", "fpf",
        "old_position", "new_position"}.
    """

    # -------------------------------------------------------------------------
    # Find the position at which we compute the figures of merit
    # -------------------------------------------------------------------------

    # If we do not optimize anything, we simply keep the initial position
    if optimize is None:
        x, y = position

    # Otherwise, we first need to find the "optimal" position
    elif optimize in ('signal', 'noise_level', 'snr', 'fpf'):

        # Define a dummy function to get the FOM which we want to optimize by
        # adjusting the position (this is essentially just verbose currying.)
        def _get_fom(pos):

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
            if optimize == 'signal':
                return -1 * signal
            if optimize == 'noise_level':
                return noise_level
            if optimize == 'snr':
                return -1 * snr
            if optimize == 'fpf':
                return fpf

        # Actually run the optimization to find the optimal position
        optimum = minimize(fun=_get_fom,
                           x0=np.array(position),
                           method="Nelder-Mead")
        x, y = np.round(optimum.x, 1)

    # Raise an error if we receive an invalid value for optimize
    else:
        raise ValueError('optimize must be either None or one of the '
                         'following: "signal", "noise_level", "snr", "fpf"!')

    # -------------------------------------------------------------------------
    # Actually compute and return the figures of merit
    # -------------------------------------------------------------------------

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
                new_position=(x, y))
