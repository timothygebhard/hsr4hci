"""
Utility functions for estimating contrasts, flux ratios and throughputs.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Optional, Tuple

from astropy.units import Quantity

import numpy as np

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
            contrast using  the "classic" approach, you can use this
            argument to pass a signal estimate that was obtained on the
            data set without  any planets. This array will be subtracted
            from the signal estimate before measuring the planet flux.
        expected_contrast: Optionally, a float containing the expected
            contrast in magnitudes. If this value is given, the
            throughput is computed (otherwise the throughput is NaN).

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
        mode='FS',
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
        mode='FS',
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
            exclusion_angle=Quantity(0, 'degree'),
        )

        # Compute the flux at the reference positions
        reference_fluxes = get_fluxes_for_polar_positions(
            polar_positions=reference_positions,
            frame=signal_estimate,
            mode='P',
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
