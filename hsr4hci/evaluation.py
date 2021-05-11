"""
Utility functions for performance evaluation (e.g., computing the SNR).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union

import math

from astropy.units import Quantity
from contexttimer.timeout import timeout
from photutils import aperture_photometry, CircularAperture
from scipy.spatial.distance import euclidean
from scipy.stats import t
from scipy.optimize import minimize, brute

import bottleneck as bn
import numpy as np

from hsr4hci.coordinates import get_center


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

def timeout_handler(*_: Any, **__: Any) -> NoReturn:
    """
    This function only serves as a callback for the @timeout decorator
    to raise a TimeoutException when the optimization takes too long.
    """
    raise TimeoutException("Optimization timed out!")


def get_number_of_apertures(
    separation: Quantity,
    aperture_radius: Quantity,
    exact: bool = True,
) -> int:
    """
    Compute the number of non-overlapping apertures with a given
    `aperture_radius` that can be placed at the given `separation`.

    Args:
        separation: The separation at which the apertures are to be
            placed (e.g., 1 lambda / D).
        aperture_radius: The radius of the apertures to be placed.
        exact: Whether to use the exact formula for the number of
            apertures, or an approximation. The latter is generally
            very good; there are only very few cases where the
            approximation over-estimates the number of apertures by 1.

    Returns:
        The number of apertures at the given separation as an integer.
    """

    # Convert the separation and the aperture radius to units of pixels (any
    # unit is fine here actually, as long as it is the same for both)
    big_r = separation.to('pixel').value
    small_r = aperture_radius.to('pixel').value

    # Sanity check: for too small separations, there are no non-overlapping
    # apertures; hence, we raise a ValueError.
    if small_r > big_r:
        raise ValueError(
            'The aperture_size must not be greater than the separation!'
        )

    # For the exact number of apertures at a given separation we need to use
    # the formula derived here: https://stackoverflow.com/a/56008236/4100721
    # Note: The additional round() call is necessary to mitigate issues due to
    # floating pointing precision. Without it, we sometimes get results like
    # "5.999999999999 apertures", which gets floored to 5 without the round().
    if exact:
        return int(math.floor(round(math.pi / math.asin(small_r / big_r), 3)))

    # Alternatively, we can use the following approximation from the Mawet et
    # al. (2014) paper, which is slightly faster to compute:
    return int(math.floor(math.pi * big_r / small_r))


def get_aperture_flux(
    frame: np.ndarray,
    position: Union[Tuple[float, float], List[Tuple[float, float]]],
    aperture_radius: Quantity,
) -> Union[float, np.ndarray]:
    """
    Get the integrated flux in an aperture (or multiple apertures) of
    the given size (i.e., `aperture_radius`) at the given `position(s)`.

    This function is essentially a convenience wrapper that bundles
    together `CircularAperture` and `aperture_photometry()`.

    Args:
        frame: A 2D numpy array of shape `(width, height)` containing
            the data on which to run the aperture photometry.
        position: A tuple `(x, y)` (or a list of such tuples) specifying
            the position(s) at which we place the aperture(s) on the
            `frame`.
        aperture_radius: The radius of the aperture. In case multiple
            positions are given, the same radius will be used for all
            of them.

    Returns:
        The integrated flux (= sum of all pixels) in the aperture.
    """

    # Create an aperture (or a set of apertures) at the given `position`
    aperture = CircularAperture(
        positions=position, r=aperture_radius.to('pixel').value
    )

    # Perform the aperture photometry, select the results
    photometry_table = aperture_photometry(frame, aperture, method='exact')
    results = np.array(photometry_table['aperture_sum'].data)

    # If there is only a single result (i.e., only a single position), we
    # can cast the result to float
    if len(results) == 1:
        return float(results)
    return results


def get_reference_aperture_positions(
    frame_size: Tuple[int, int],
    position: Tuple[float, float],
    aperture_radius: Quantity,
    ignore_neighbors: int = 1,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Compute the positions of the reference apertures (i.e., the
    apertures used to estimate the noise level) for a signal aperture
    that is placed at `position`.

    Args:
        frame_size: A tuple of integers `(width, height)` specifying the
            size of the frames that we are working with. (Necessary to
            compute the (Cartesian) coordinates of the apertures.)
        position: A tuple `(x, y)` specifying the position at which the
            signal aperture will be placed. Usually, this is the
            suspected of a planet.
        aperture_radius: The radius of the apertures to be placed.
        ignore_neighbors: The number of neighboring apertures that will
            *not* be used as reference positions.
            Rationale: methods like PCA often cause visible negative
            self-subtraction "wings" left and right of the planet
            signal. As these do not provide an unbiased estimate of the
            background noise, we usually want to exclude them from the
            reference positions.

    Returns:
        A tuple `(reference, ignored)` where both `reference` and
        `ignored` are a list of tuples `(x, y)` with the positions of
        the actual reference apertures as well as the positions of the
        apertures that are ignored because of `ignore_neighbors`. If
        the latter is 0, the `ignored` list will be empty.
    """

    # Compute the frame center and polar representation of initial position
    center = get_center(frame_size=frame_size)
    rho = math.sqrt(
        (position[0] - center[0]) ** 2 + (position[1] - center[1]) ** 2
    )
    phi = math.atan2(position[1] - center[1], position[0] - center[0])

    # Compute the total number of apertures that can be placed at the
    # separation of the given `position`.This also includes the signal
    # aperture at `position`, which is why we will need to subtract 1.
    n_apertures = get_number_of_apertures(
        separation=Quantity(rho, 'pixel'), aperture_radius=aperture_radius
    )
    n_apertures -= 1

    # Collect the indices of the positions which are ignored (namely, the
    # first and last `ignore_neighbors`)
    ignored_idx = np.full(n_apertures, False)
    ignored_idx[:ignore_neighbors] = True
    ignored_idx = np.logical_or(ignored_idx, ignored_idx[::-1])

    # Get angles at which to place the apertures. The `+ 1` is needed to take
    # into account the position of the signal aperture.
    angles = np.linspace(0, 2 * np.pi, n_apertures + 1, endpoint=False)[1:]
    angles = np.mod(angles + phi, 2 * np.pi)

    # Compute x- and y-positions in Cartesian coordinates
    x_positions = rho * np.cos(angles) + center[0]
    y_positions = rho * np.sin(angles) + center[1]

    # Split the positions into reference and ignored positions and convert
    # them into lists of tuples
    reference = [
        (float(x), float(y))
        for x, y in zip(x_positions[~ignored_idx], y_positions[~ignored_idx])
    ]
    ignored = [
        (float(x), float(y))
        for x, y in zip(x_positions[ignored_idx], y_positions[ignored_idx])
    ]

    return reference, ignored


def compute_snr(
    frame: np.ndarray,
    position: Tuple[float, float],
    aperture_radius: Quantity,
    ignore_neighbors: int,
) -> Tuple[float, float, float, float]:
    """
    Compute the signal-to-noise ratio (SNR) and related performance
    metrics (such as the FPF) for a given position in an image.

    For more detailed information about the definition of the SNR and
    the motivation behind it, see the following paper:

        Mawet, D. et al. (2014): "Fundamental limitations of high
            contrast imaging set by small sample statistics". *The
            Astrophysical Journal*, 792(2), 97.
            DOI: 10.1088/0004-637X/792/2/97

    The implementation in this function is strongly inspired by the
    respective PynPoint method that implements this functionality,
    namely `pynpoint.utils.analysis.false_alarm`.

    Args:
        frame: A 2D numpy array of shape (width, height) containing the
            input frame (i.e., e.g., derotated and merged residuals).
        position: A tuple (x, y) specifying the position at which to
            compute the SNR and related quantities.
        aperture_radius: The radius of the apertures to be used. This
            value is commonly chosen as 0.5 * lambda / D.
        ignore_neighbors: The number of neighboring apertures that will
            *not* be used as reference positions. Rationale: methods
            like PCA often cause visible negative self-subtraction
            "wings" left and right of the planet signal. As these do not
            provide an unbiased estimate of the background noise, we
            usually want to exclude them from the reference positions.

    Returns:
        A four-tuple consisting of the following four figures of merit:

            signal: The numerator of the SNR, that is, the integrated
                flux inside the signal aperture minus the mean of the
                noise apertures.
            noise: The denominator of the SNR, that is, the standard
                deviation of the integrated flux of the noise apertures,
                times a correction factor for small sample statistics.
            snr: The signal-to-noise ratio (SNR) as defined in eq. (8)
                of Mawet et al. (2014).
            fpf: The false positive fraction (FPF) as defined in
                eq. (10) of Mawet et al. (2014).
    """

    # Get the positions of the reference apertures, that is, the positions at
    # which we will measure the flux to estimate the noise level
    reference_positions, _ = get_reference_aperture_positions(
        frame_size=(frame.shape[0], frame.shape[1]),
        position=position,
        aperture_radius=aperture_radius,
        ignore_neighbors=ignore_neighbors,
    )
    n_apertures = len(reference_positions)

    # Make sure we have have enough reference positions to compute the FPF
    if n_apertures < 2:
        raise ValueError(
            f'Number of reference apertures is too small to calculate '
            f'the FPF! (n = {n_apertures})'
        )

    # Get the integrated flux in all the reference apertures
    reference_fluxes = get_aperture_flux(
        frame=frame,
        position=reference_positions,
        aperture_radius=aperture_radius,
    )

    # Get the integrated flux in the signal aperture
    signal_flux = float(
        get_aperture_flux(
            frame=frame, position=position, aperture_radius=aperture_radius
        )
    )

    # Compute the "signal", that is, the numerator of the signal-to-noise
    # ratio: According to eq. (8) in Mawet et al. (2014), this is given by
    # the difference between the integrated flux in the signal aperture and
    # the mean of the integrated flux in the reference apertures
    signal = signal_flux - bn.nanmean(reference_fluxes)

    # Compute the "noise", that is, the denominator of the signal-to-noise
    # ratio: According to eq. (8) in Mawet et al. (2014), this is given by
    # the *unbiased* standard deviation (i.e., including Bessel's correction)
    # of the integrated flux in the reference apertures times a correction
    # factor to account for the small sample statistics.
    noise = bn.nanstd(reference_fluxes, ddof=1) * np.sqrt(1 + 1 / n_apertures)

    # Compute the SNR by dividing the "signal" through the "noise"
    snr = signal / noise

    # Compute the false positive fraction (FPF). According to eq. (10) in
    # Mawet et al. (2014), the FPF is given by 1 - F_nu(SNR), where F_nu is
    # the cumulative distribution function (CDF) of a t-distribution with
    # `nu = n-1` degrees of freedom, where n is the number of reference
    # apertures. For numerical reasons, we use the survival function (SF),
    # which is defined precisely as 1-CDF, but may give more accurate results.
    fpf = t.sf(snr, df=(n_apertures - 1))

    return signal_flux, noise, snr, fpf


def compute_optimized_snr(
    frame: np.ndarray,
    position: Tuple[float, float],
    aperture_radius: Quantity,
    ignore_neighbors: int = 0,
    target: Optional[str] = 'signal_flux',
    max_distance: Optional[float] = 1.0,
    method: str = 'brute',
    grid_size: int = 16,
    time_limit: int = 30,
) -> Dict[str, Any]:
    """
    Compute the *optimized* signal-to-noise ratio (SNR) and associated
    quantities (false positive fraction, signal sum, noise).

    This function is only a wrapper around the `compute_snr()` function.
    It simply encapsulates all the code that is needed for the various
    optimization options.

    Args:
        frame: A 2D numpy array of shape (width, height) containing the
            input frame (i.e., e.g., derotated and merged residuals).
        position: A tuple (x, y) specifying the position at which to
            compute the SNR and related quantities. If `optimize` is
            not `None`, this position is only used as the starting point
            for the optimization.
        aperture_radius: The radius of the apertures to be used. This
            value is commonly chosen as 0.5 * lambda / D.
        ignore_neighbors: The number of neighboring apertures that will
            *not* be used as reference positions. Rationale: methods
            like PCA often cause visible negative self-subtraction
            "wings" left and right of the planet signal. As these do not
            provide an unbiased estimate of the background noise, we
            usually want to exclude them from the reference positions.
            Default is 0 (i.e., do not ignore any apertures).
        target: Either None, or a string containing the target quantity
            that will be optimized by the optimization method by varying
            the planet position. Choices are the following:
                None: Do not perform any optimization and simply compute
                    the SNR at the given position.
                "signal_flux": The flux of the signal aperture (which
                    is completely independent of all noise apertures).
                    This is the default.
                "signal": The numerator of the SNR, that is, the
                    integrated flux inside the signal aperture minus
                    the mean of the noise apertures.
                "noise": The denominator of the SNR, that is, the
                    standard deviation of the integrated flux of the
                    noise apertures, times a correction factor for small
                    sample statistics.
                "snr": The signal-to-noise ratio (SNR) as defined in
                    eq. (8) of Mawet et al. (2014).
                "fpf": The false positive fraction (FPF) as defined in
                    eq. (10) of Mawet et al. (2014).
        max_distance: When using an optimizer, this parameter controls
            the maximum (Euclidean) distance of the optimal position
            from the initial position (in pixels).
            This is particularly useful for faint sources, where we do
            not want the optimizer to just "wander off" too far from the
            initial position. If set to None, there is no such maximum
            distance. Default is 1 pixel.
        method: A string containing the optimization method to be used.
            This must either be "brute" for brute-force optimization
            via a grid search, or an optimization method supported by
            scipy.optimize.minimize(); see the scipy docs for more
            details. Default is "brute".
        grid_size: A positive integer specifying the size of the grid
            used for brute-force optimization. All other optimization
            methods ignore this parameter. Default is 16.
        time_limit: An integer specifying the maximum runtime of the
            function in seconds. This is particularly useful when using
            an optimizer which may get stuck. Default is 30 seconds.

    Returns:
        A dictionary containing the following keys:

            signal: The signal (i.e., the integrated flux in the signal
                aperture minus the mean flux in the reference apertures)
                at the optimized position.
            noise: The noise (i.e., the unbiased standard deviation of
                the reference apertures, times a correction factor for
                taking into account small sample statistics) at the
                optimized position.
            snr: The signal-to-noise ratio at the optimized position.
            fpf: The false positive fraction at the optimized position.
            old_position: The starting position for the optimizer, i.e.,
                the original `position` argument.
            new_position: The final position found by the optimizer.
            message: The final status message of the optimizer. Might
                contain, for example, a convergence warning.
            success: The final status of the optimizer.
    """

    # -------------------------------------------------------------------------
    # Run some basic sanity checks on the parameters we have received
    # -------------------------------------------------------------------------

    assert isinstance(frame, np.ndarray), 'frame must be a numpy array!'
    assert frame.ndim == 2, 'frame must be 2-dimensional!'
    assert aperture_radius > 0, 'aperture_radius must be positive!'
    assert ignore_neighbors >= 0, 'ignore_neighbors must be non-negative!'
    assert grid_size > 0, 'grid_size must be positive!'
    assert time_limit > 0, 'time_limit must be positive!'

    if max_distance is None:
        max_distance = np.inf
    else:
        assert max_distance > 0, 'max_distance must be positive or None!'

    # -------------------------------------------------------------------------
    # Define dummy function and wrap it with a @timeout decorator
    # -------------------------------------------------------------------------

    # We would like the time limit for this function to be definable as an
    # argument. Since the time limit is enforced using the @timeout decorator,
    # we therefore need to define a dummy function which encapsulates all the
    # real functionality, and which we can then decorate with the corresponding
    # timeout decorator (which takes the desired time limit as an argument).

    @timeout(limit=int(time_limit), handler=timeout_handler)
    def _compute_optimized_snr(
        frame: np.ndarray,
        position: Tuple[float, float],
        aperture_radius: Quantity,
        ignore_neighbors: int = 1,
        target: Optional[str] = None,
        max_distance: float = 1.0,
        method: str = 'brute',
        grid_size: int = 16,
    ) -> Dict[str, Any]:
        """
        This is a dummy function which is needed to limit the runtime.
        """

        # ---------------------------------------------------------------------
        # Find the position at which we compute the figures of merit
        # ---------------------------------------------------------------------

        # If we do not optimize anything, we simply keep the initial position
        # and use default value for the "optimization" success and message
        if target is None:
            x, y = position
            message = 'Figure of merit computation terminated successfully.'
            success = True

        # Otherwise, we first need to find the "optimal" position
        elif target in ('signal_flux', 'signal', 'noise', 'snr', 'fpf'):

            # -----------------------------------------------------------------
            # Define an objective function for the optimizer
            # -----------------------------------------------------------------

            def objective_func(
                candidate_position: Tuple[float, float],
            ) -> float:

                # Initialize default values, which indicate that the current
                # candidate position is either not admissible, or not optimal
                signal_flux, signal, noise, snr, fpf = (
                    -np.inf,
                    -np.inf,
                    np.inf,
                    -np.inf,
                    np.inf,
                )

                # Compute the distance between the candidate_position and the
                # initial position. If this value is larger than the given
                # max_distance, we return the above default values to indicate
                # that the position is not admissible.
                # Only if the candidate_position is close enough to the initial
                # position do we actually compute the signal_flux or the SNR.
                distance = euclidean(position, candidate_position)

                if (max_distance is not None) or (distance <= max_distance):

                    # Check if we only need the signal flux (this is cheaper
                    # to compute than the SNR)
                    if target == 'signal_flux':

                        signal_flux = float(
                            get_aperture_flux(
                                frame=frame,
                                position=candidate_position,
                                aperture_radius=aperture_radius,
                            )
                        )

                    else:

                        # Try to compute the SNR at the current position. If
                        # this fails for some reason, use default values to
                        # indicate that the current position is not the optimum
                        try:
                            signal, noise, snr, fpf = compute_snr(
                                frame=frame,
                                position=candidate_position,
                                aperture_radius=aperture_radius,
                                ignore_neighbors=ignore_neighbors,
                            )
                        except ValueError:
                            signal, noise, snr, fpf = (
                                -np.inf,
                                np.inf,
                                -np.inf,
                                np.inf,
                            )

                # Depending on the target, we either want to minimize x or -x.
                # For the FPF, we optimize the *negative inverse*, as the FPF
                # can get very small, which is a problem for some optimizers.
                if target == 'signal_flux':
                    return float(-1 * signal_flux)
                if target == 'signal':
                    return float(-1 * signal)
                if target == 'noise':
                    return float(noise)
                if target == 'snr':
                    return float(-1 * snr)
                if target == 'fpf':
                    return float(-1 / fpf)
                raise ValueError('Invalid value for "target"!')

            # -----------------------------------------------------------------
            # Run optimizer to minimize the objective function
            # -----------------------------------------------------------------

            # Depending on the chosen optimization method, we need to use
            # different functions with different signatures, which is why
            # we need the following case analysis:

            # Option 1: Brute-force optimization over a grid
            if method == 'brute':

                # Run the optimizer
                # NOTE: The finish=None argument is necessary so that
                # the optimizer does not also run a downhill-simplex
                # optimization starting at the optimal grid point!
                optimum = brute(
                    func=objective_func,
                    ranges=(
                        (
                            position[0] - max_distance,
                            position[0] + max_distance,
                        ),
                        (
                            position[1] - max_distance,
                            position[1] + max_distance,
                        ),
                    ),
                    Ns=grid_size,
                    finish=None,
                )

                # Get the result and set up the message and status
                x, y = tuple(np.round(optimum, 2))
                message = 'Brute-force optimization finished.'
                success = True

            # Option 2: "Regular" optimization (e.g., Nelder-Mead)
            else:

                # Run the optimizer
                optimum = minimize(
                    fun=objective_func, x0=np.array(position), method=method
                )

                # Get the result and set up the message and status
                x, y = tuple(np.round(optimum.x, 2))
                message = optimum.message
                success = optimum.success

        # Raise an error if we receive an invalid value for optimize
        else:
            raise ValueError(
                'optimize must be one of the following: '
                '[None, "signal_flux", "signal", "noise", "snr", "fpf"]!'
            )

        # ---------------------------------------------------------------------
        # Actually compute SNR at the optimal position and return results
        # ---------------------------------------------------------------------

        # Compute signal, noise, SNR and FPF at the optimal position
        signal, noise, snr, fpf = compute_snr(
            frame=frame,
            position=(x, y),
            aperture_radius=aperture_radius,
            ignore_neighbors=ignore_neighbors,
        )

        # Return everything, including the position found by the optimizer
        return dict(
            signal=signal,
            noise=noise,
            snr=snr,
            fpf=fpf,
            old_position=position,
            new_position=(x, y),
            message=message,
            success=success,
        )

    # -------------------------------------------------------------------------
    # Run dummy function to optimize the target quantity with a time limit
    # -------------------------------------------------------------------------

    # Finally, we simply call the dummy function with the arguments passed to
    # this function. It it finished successfully, we simply return its output:
    try:

        result: Dict[str, Any] = _compute_optimized_snr(
            frame=frame,
            position=position,
            aperture_radius=aperture_radius,
            ignore_neighbors=ignore_neighbors,
            target=target,
            method=method,
            grid_size=grid_size,
            max_distance=max_distance,
        )

    # If the computation fails with a timeout, we return default values:
    except TimeoutException:

        result = dict(
            signal=None,
            noise=None,
            snr=None,
            fpf=None,
            old_position=position,
            new_position=None,
            message='Optimization timed out!',
            success=False,
        )

    return result
