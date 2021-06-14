"""
Utilities for computing performance metrics (e.g., SNR).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any, Dict, List, Optional, Tuple, Union

from astropy.units import Quantity

import numpy as np
import scipy.stats as stats

from hsr4hci.coordinates import cartesian2polar, polar2cartesian
from hsr4hci.photometry import get_flux, get_fluxes_for_polar_positions
from hsr4hci.positions import (
    get_reference_positions,
    rotate_reference_positions,
)


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def two_sample_t_test(
    planet_samples: Union[List[float], np.ndarray],
    noise_samples: Union[List[float], np.ndarray],
) -> Tuple[float, float, float, float, float]:
    """
    Compute the two-sample t-test that is the basis of the signal-to-
    noise (SNR) as introduced by the following classic paper:

        Mawet, D. et al. (2014): "Fundamental limitations of high
            contrast imaging set by small sample statistics". *The
            Astrophysical Journal*, 792(2), 97.
            DOI: 10.1088/0004-637X/792/2/97

    Args:
        planet_samples: A list of floats containing the results of the
            flux measurements at the planet position(s). Generally, in
            almost all cases, there is only a single planet position
            and, therefore, only a single planet planet sample.
        noise_samples: A list of floats containing the results of the
            flux measurements at the reference (or noise) positions.

    Returns:
        A 5-tuple consisting of the following values:
            (signal, noise, snr, fpf, p_value)
        The signal and noise are the numerator and denominator of the
        SNR, which itself is the test statistic of the t-test that is
        being performed by this function. The false positive fraction
        (FPF) and the p-value are directly derived from the SNR under
        the assumption of a t-distribution for the SNR.
    """

    # Determine the number of samples; generally, for computing the SNR, there
    # will only be a single planet aperture, that is, n_1 = 1
    n_1 = len(planet_samples)
    n_2 = len(noise_samples)

    # Sanity checks
    if n_1 < 1:
        raise ValueError('planet_samples must have at least 1 entry!')
    if n_2 < 2:
        raise ValueError('noise_samples must have at least 2 entries!')

    # Compute the mean of the planet samples (generally, this is just the one
    # planet sample we have), and the mean of the noise / reference apertures
    mean_planet = float(np.mean(planet_samples))
    mean_noise = float(np.mean(noise_samples))

    # Compute the "signal" (= the numerator of the signal-to-noise ratio).
    # According to eq. (8) in Mawet et al. (2014), this is given by the
    # difference between the (integrated) flux at the "planet position" and
    # the mean of the (integrated) fluxes at the reference positions.
    signal = mean_planet - mean_noise

    # Compute the "noise" (= the denominator of the signal-to-noise ratio).
    # According to eq. (8) in Mawet et al. (2014), this is given by the
    # *unbiased* standard deviation (i.e., including Bessel's correction) of
    # the (integrated) flux in the reference apertures times a correction
    # factor to account for the small sample statistics.
    noise = np.std(noise_samples, ddof=1) * np.sqrt(1 / n_1 + 1 / n_2)

    # Compute the SNR. The SNR is the test statistic of the two-sample t-test,
    # and it should follow a t-distribution with a number of degrees of freedom
    # that depends on the number of samples (see below).
    snr = signal / noise

    # The number of degrees of freedom is given by the number of samples
    df = n_1 + n_2 - 2

    # Compute the false positive fraction (FPF) and the p-value. Unlike the
    # SNR, these can be compared "universally", because they do not depend on
    # the position (or more precisely: the number of reference positions that
    # is associated with a position) anymore.
    # According to eq. (10) in Mawet et al. (2014), the FPF is given by
    # 1 - F_nu(SNR), where F_nu is the cumulative distribution function (CDF)
    # of a t-distribution with `nu = n-1` degrees of freedom, where n is the
    # number of reference apertures. For numerical reasons, we use the survival
    # function (SF), which is defined precisely as 1-CDF, but may give more
    # accurate results.
    fpf = stats.t.sf(snr, df=df)
    p_value = stats.t.cdf(snr, df=df)

    return signal, noise, snr, fpf, p_value


def compute_metrics(
    frame: np.ndarray,
    polar_position: Tuple[Quantity, Quantity],
    aperture_radius: Quantity,
    planet_mode: str = 'FS',
    noise_mode: str = 'P',
    search_radius: Optional[Quantity] = Quantity(1, 'pixel'),
    exclusion_angle: Optional[Quantity] = None,
    n_rotation_steps: int = 100,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]]]:
    """
    Compute evaluation metrics (SNR, FPF, ...) at a given position.

    Args:
        frame: The frame (usually a signal estimate) on which to compute
            the metrics.
        polar_position: The position of the (candidate) planet as a
            2-tuple `(separation, angle)` using "astronomical" polar
            coordinates (i.e., 0 degrees = North = "up", not "right",
            as in mathematical polar coordinates).
        aperture_radius: If the planet / noise mode is aperture-based,
            this parameter controls the size of the apertures.
            Regardless of the mode, this value is required to determine
            the number of reference positions; therefore it cannot be
            optional. (Usually set this to 1/2 of the FWHM of the PSF.)
        planet_mode: The `mode` to be used to measure the flux of the
            planet / signal. See `hsr4hci.photometry.get_flux()` for
            more details.
        noise_mode: The `mode` to be used to measure the flux at the
            reference positions. See `hsr4hci.photometry.get_flux()`
            for more details. Note that this should be compatible with
            the choice for the `planet_mode`; i.e., if the mode for the
            planet is "FS", the mode for the noise should be "P", and
            if the planet mode is "ASS", the noise mode should be "AS".
        search_radius: If the planet mode is search-based (mode "ASS"
            or "FS"), this parameter controls how big the area is that
            should be considered for maximizing the planet flux.
        exclusion_angle: This parameter controls how the reference
            positions are chosen. It can be used, for example, to
            exclude the reference positions immediately to the left and
            right of the planet position, because for some algorithms
            (e.g., PCA), these are known to contain self-subtraction /
            over-subtraction "wings" which do not give an unbiased
            estimate of the background. For more details, see
            `hsr4hci.positions.get_reference_positions()`.
        n_rotation_steps: This parameter determines the number of
            rotation steps that are applied to the reference positions:
            The exact placement of the reference positions is always
            somewhat arbitrary, but can have a rather large effect on
            the final metrics. By rotating the reference positions, we
            can at least get a feeling for the size of the effect. See
            `hsr4hci.positions.rotate_reference_positions()` for more.
            If this value is set to 0, no rotations are performed.

    Returns:
        A 2-tuple, consisting of (1) a (nested) dictionary containing
        the mean, median, standard deviation, minimum and maximum of
        each metric, and (2) the position of the planet before and
        after a potential optimization.
    """

    # Define a shortcut for the frame size
    frame_size = (frame.shape[0], frame.shape[1])

    # Compute initial position in Cartesian coordinates
    initial_position_cartesian = polar2cartesian(
        separation=polar_position[0],
        angle=polar_position[1],
        frame_size=frame_size,
    )

    # Measure the planet flux and get its final (= optimized) position; both
    # in Cartesian and in polar coordinates
    final_position_cartesian, planet_flux = get_flux(
        frame=frame,
        position=initial_position_cartesian,
        mode=planet_mode,
        aperture_radius=aperture_radius,
        search_radius=search_radius,
    )
    final_position_polar = cartesian2polar(
        position=final_position_cartesian, frame_size=frame_size
    )

    # Collect the planet positions before and after a potential optimization,
    # both in Cartesian and (astronomical) polar coordinates
    positions = {
        'final': {
            'polar': final_position_polar,
            'cartesian': final_position_cartesian,
        },
        'initial': {
            'polar': polar_position,
            'cartesian': initial_position_cartesian,
        },
    }

    # Get the reference positions for the final planet position
    reference_positions = get_reference_positions(
        polar_position=final_position_polar,
        aperture_radius=aperture_radius,
        exclusion_angle=exclusion_angle,
    )

    # Check that we have enough reference positions to continue computation
    if len(reference_positions) < 2:
        raise RuntimeError('Too few reference positions (i.e., < 2)!')

    # Create rotated versions of the reference positions so that we can
    # estimate how much the final metrics depend on the exact placement of
    # the reference positions (which is, to some degree, arbitrary).
    rotated_reference_positions = rotate_reference_positions(
        reference_positions=reference_positions,
        n_steps=n_rotation_steps,
    )

    # Keep track of the result variables for the t-test(s)
    signals = []
    noises = []
    snrs = []
    fpfs = []
    log_fpfs = []
    p_values = []

    # Loop over the different reference positions, measure the fluxes at
    # the (rotated) reference positions, and perform a two-sample t-test to
    # compute the respective metrics (SNR, FPF, ...)
    for polar_positions in rotated_reference_positions:

        # Compute the fluxes at the rotated reference positions
        noise_samples = get_fluxes_for_polar_positions(
            polar_positions=polar_positions,
            frame=frame,
            mode=noise_mode,
            aperture_radius=aperture_radius,
            search_radius=None,
        )

        # Compute the two-sample t-test; store the results
        signal, noise, snr, fpf, p_value = two_sample_t_test(
            planet_samples=[planet_flux], noise_samples=noise_samples
        )
        signals.append(signal)
        noises.append(noise)
        snrs.append(snr)
        fpfs.append(fpf)
        log_fpfs.append(-np.log10(fpf))
        p_values.append(p_value)

    # Construct results dictionary by looping over all combinations of result
    # quantities and aggregation functions
    results: Dict[str, Dict[str, float]] = {}
    for metric_name, metric_values in (
        ('signal', signals),
        ('noise', noises),
        ('snr', snrs),
        ('fpf', fpfs),
        ('log_fpf', log_fpfs),
        ('p_value', p_values),
    ):
        for aggregation_function in (
            np.nanmean,
            np.nanmedian,
            np.nanstd,
            np.nanmin,
            np.nanmax,
        ):

            if metric_name not in results.keys():
                results[metric_name] = {}

            name = aggregation_function.__name__.replace('nan', '')
            results[metric_name][name] = float(
                aggregation_function(metric_values)
            )

    return results, positions
