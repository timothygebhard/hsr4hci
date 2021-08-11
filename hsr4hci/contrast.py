"""
Utility functions for estimating contrasts, flux ratios and throughputs,
as well as computing contrast curves (i.e., detection limits).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Optional, Tuple

from astropy.units import Quantity
from scipy.interpolate import CubicSpline
from scipy.stats import norm

import numpy as np
import pandas as pd

from hsr4hci.coordinates import cartesian2polar, polar2cartesian
from hsr4hci.photometry import (
    get_stellar_flux,
    get_flux,
    get_fluxes_for_polar_positions,
)
from hsr4hci.positions import get_reference_positions
from hsr4hci.psf import get_psf_fwhm
from hsr4hci.units import magnitude_to_flux_ratio, flux_ratio_to_magnitudes


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_contrast(
    signal_estimate: np.ndarray,
    polar_position: Tuple[Quantity, Quantity],
    psf_template: np.ndarray,
    metadata: dict,
    no_fake_planets: Optional[np.ndarray] = None,
    expected_contrast: Optional[float] = None,
    planet_mode: str = 'FS',
    noise_mode: str = 'P',
    exclusion_angle: Optional[Quantity] = None,
) -> dict:
    """
    Compute the contrast and flux ratio for the planet at the given
    `polar_position` in the `signal_estimate`, and, if desired, also
    compute the throughput (i.e., the ratio of the observed and the
    expected flux ratio).

    Args:
        signal_estimate: A 2D numpy array with the signal estimate.
        polar_position: A 2-tuple of Quantities, (separation, angle),
            specifying the position of the planet for which to compute
            the contrast and flux ratio.
        psf_template: A 2D numpy array containing the *unsaturated* (!)
            and *unnormalized* (!) PSF template for the data set on
            which the `signal_estimate` was obtained.
        metadata: A dictionary containing metadata about the data set.
            Requires the following keys to compute the stellar flux:
            "DIT_STACK", "DIT_PSF_TEMPLATE" and "ND_FILTER".
        no_fake_planets: Optionally, a 2D numpy array with the same
            shape as the `signal_estimate` If you want to compute the
            contrast using the "classic" approach, you can use this
            argument to pass a signal estimate that was obtained on the
            data set without any planets. This array will be subtracted
            from the signal estimate before measuring the planet flux.
        expected_contrast: Optionally, a float containing the expected
            contrast in magnitudes. If this value is given, the
            throughput is computed (otherwise the throughput is NaN).
        planet_mode: Photometry mode that is used to measure the planet
            flux. See `hsr4hci.photometry.get_flux()` for details.
        noise_mode: Photometry mode that is used to measure the flux at
            the reference positions. See `hsr4hci.photometry.get_flux()`
            for details.
        exclusion_angle: Exclusion angle that is used for determining
            the reference positions (basically: whether or not to ignore
            the positions left and right of the `polar_position` which
            may contain self-subtraction "wings"). For more details,
            see `hsr4hci.positions.get_reference_positions()`. This
            option is only used if `no_fake_planets` is None.

    Returns:
        A dictionary containing the observed contrast and flux ratio,
        the expected contrast and flux ratio, the throughput, the raw
        flux and the background flux, the stellar flux, and the
        (optimized) Cartesian position of the planet candidate.
    """

    # Define shortcuts; convert polar position to Cartesian one
    frame_size = (signal_estimate.shape[0], signal_estimate.shape[1])
    cartesian_position = polar2cartesian(
        *polar_position, frame_size=frame_size
    )

    # -------------------------------------------------------------------------
    # Compute the PSF FWHM and the stellar flux
    # -------------------------------------------------------------------------

    # Get FWHM of the PSF template
    psf_fwhm = get_psf_fwhm(psf_template=psf_template)

    # Compute the stellar flux
    stellar_flux = get_stellar_flux(
        psf_template=psf_template,
        dit_stack=metadata['DIT_STACK'],
        dit_psf_template=metadata['DIT_PSF_TEMPLATE'],
        scaling_factor=metadata['ND_FILTER'],
        mode=planet_mode,
        aperture_radius=Quantity(psf_fwhm / 2, 'pixel'),
        search_radius=Quantity(1, 'pixel'),
    )

    # -------------------------------------------------------------------------
    # Compute flux at the target position and the reference positions
    # -------------------------------------------------------------------------

    # Compute the frame on which to perform the photometry. Classically, we
    # subtract the residual that was obtained for the "no artificial planets
    # injected" case from our signal estimate (to estimate the background) and
    # run the photometry.
    # In practice, we might not always have such a "no_fake_planets" residual;
    # in these cases, we estimate the background from reference positions at
    # the same separation (the same ones that we use for computing the SNR).
    if no_fake_planets is None:
        frame = np.nan_to_num(signal_estimate, copy=True)
    else:
        frame = np.nan_to_num(signal_estimate - no_fake_planets, copy=True)

    # Compute the flux at this position
    final_position_cartesian, raw_flux = get_flux(
        frame=frame,
        position=cartesian_position,
        mode=planet_mode,
        aperture_radius=Quantity(psf_fwhm / 2, 'pixel'),
        search_radius=Quantity(1, 'pixel'),
    )

    # Convert the optimized position to polar coordinates
    final_position_polar = cartesian2polar(
        position=final_position_cartesian, frame_size=frame_size
    )

    # If no "no_fake_planets" residual was given, we
    if no_fake_planets is None:

        # Compute reference positions for the final position
        reference_positions = get_reference_positions(
            polar_position=final_position_polar,
            aperture_radius=Quantity(psf_fwhm / 2, 'pixel'),
            exclusion_angle=exclusion_angle,
        )

        # Compute the flux at the reference positions
        reference_fluxes = get_fluxes_for_polar_positions(
            polar_positions=reference_positions,
            frame=signal_estimate,
            mode=noise_mode,
        )

        # Estimate the background flux by averaging the reference fluxes
        background_flux = np.median(reference_fluxes)

    else:
        background_flux = 0

    # Subtract the background from the raw flux
    flux = raw_flux - background_flux

    # -------------------------------------------------------------------------
    # Compute the flux ratio, the contrast, and the throughput (if applicable)
    # -------------------------------------------------------------------------

    # Compute the *observed* flux ratio and contrast
    observed_flux_ratio = flux / stellar_flux
    if observed_flux_ratio > 0 and not np.isclose(observed_flux_ratio, 0):
        observed_contrast = flux_ratio_to_magnitudes(observed_flux_ratio)
    else:
        observed_contrast = np.infty

    # Compute the *expected* flux ratio (if applicable)
    if expected_contrast is not None:
        expected_flux_ratio = magnitude_to_flux_ratio(expected_contrast)
    else:
        expected_flux_ratio = np.nan

    # Compute the throughput (if applicable). The throughput is the ratio of
    # the observed and the expected flux ratio.
    if expected_contrast is not None:
        throughput = max(observed_flux_ratio / expected_flux_ratio, 0)
    else:
        throughput = np.nan

    # -------------------------------------------------------------------------
    # Construct and return the final results dictionary
    # -------------------------------------------------------------------------

    return dict(
        observed_flux_ratio=observed_flux_ratio,
        observed_contrast=observed_contrast,
        expected_flux_ratio=expected_flux_ratio,
        expected_contrast=expected_contrast,
        throughput=throughput,
        stellar_flux=stellar_flux,
        raw_flux=raw_flux,
        background_flux=background_flux,
        final_position_cartesian=final_position_cartesian,
    )


def get_contrast_curve(
    df: pd.DataFrame,
    sigma_threshold: float = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a data frame `df` with experiment results, compute a contrast
    curve, that is, for each separation, determine the contrast value
    until which we can detect a planet with a confidence that matches
    the given `sigma_threshold`.

    Args:
        df: A pandas data frame with experiment results. In particular,
            we need columns for the contrast, the separation and the
            false positive fraction (FPF).
        sigma_threshold: The significance threshold for what we still
            want to accept as "detectable". The usual value of 5 sigma
            (based on a standard normal distribution) corresponds to a
            1 in 3.5 million chance of a false positive.

    Returns:
        A 2-tuple, (separations, detection_limits), which contains the
        detection limit for each separation.
    """

    # Convert the target sigma threshold into a threshold value for the FPF
    fpf_threshold = 1 - norm.cdf(sigma_threshold, 0, 1)

    # Get the separation and contrast values for which we have results
    separations = np.array(sorted(np.unique(df.separation.values)))
    expected_contrasts = sorted(np.unique(df.expected_contrast.values))

    # Store the detection limits (i.e., the threshold contrast value below
    # which we can no longer detect the planet reliably) for each separation
    detection_limits = np.full_like(separations, np.nan, dtype=np.float64)

    # Loop over the separation values and compute the detection limits
    for i, separation in enumerate(separations):

        # Initialize lists in which we collect values for interpolator
        average_fpf_values = []

        # Loop over expected contrasts for which we have results
        for expected_contrast in expected_contrasts:

            # Select subset of the data frame that matches the separation and
            # the expected contrast. This should usually contain 6 entries for
            # the six azimuthal positions.
            df_subset = df[
                (df.separation == separation)
                & (df.expected_contrast == expected_contrast)
            ]

            # Compute the average FPF for the current combination of the
            # separation and the expected contrast
            average_fpf = np.mean(df_subset['fpf_mean'].values)

            # Store the expected contrast and the average FPF
            average_fpf_values.append(average_fpf)

        # Apply a cubic spline interpolation to the curve of expected contrast
        # values and average FPF values, and find the point where this curve
        # takes on the value of the desired `fpf_threshold`. This point (if it
        # exists) is then stored as the value of the contrast curve for the
        # current separation.
        interpolator = CubicSpline(
            x=expected_contrasts,
            y=np.array(average_fpf_values) - fpf_threshold,
            extrapolate=False,
        )

        # If multiple threshold points exist, store the maximum
        if len(interpolator.roots()) > 0:
            detection_limits[i] = max(interpolator.roots())

    return separations, detection_limits
