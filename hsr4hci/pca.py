"""
Utility functions for performing principal component analysis (PCA).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from copy import deepcopy
from typing import Iterable, Optional, Tuple, Union
from warnings import warn

from sklearn.decomposition import PCA

import numpy as np

from hsr4hci.derotating import derotate_combine


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_pca_noise_estimate(
    stack: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """
    Get the PCA-based estimate for the systematic noise in the given
    ``stack`` by computing a lower-dimensional approximation it. This
    is based on the assumption that the noise is responsible for most
    of the variance in the stack.

    More specifically, this function takes the given ``stack`` and uses
    principal component analysis to compute a basis of eigenimages. It
    then only keeps the first `n_components` basis vectors, projects all
    frames into this new basis (thus reducing the stack's effective
    dimensionality), and then maps this lower-dimensional approximation
    of the stack back into the original space.

    Args:
        stack: A 3D numpy array of shape `(n_frames, width, height)`
            containing the stack for which to estimate the systematic
            noise using PCA.
        n_components: The number of components for the PCA, that is,
            the effective number of dimensions of the noise estimate.

    Returns:
        A numpy array that has the same shape as the original ``stack``
        which contains the PCA-based estimate for the systematic noise
        in the stack.
    """

    # Cover a special corner case: for 0 principal components, the noise
    # estimate is 0 (alternatively, it could be the mean / median?)
    if n_components == 0:
        return np.zeros_like(stack)

    # Instantiate new PCA
    pca = PCA(n_components=n_components)

    # Reshape stack from 3D to 2D: each frame is turned into a single long
    # vector of length width * height
    reshaped_stack = stack.reshape(stack.shape[0], -1)

    # Fit PCA and apply dimensionality reduction
    transformed_stack = pca.fit_transform(reshaped_stack)

    # Use inverse transform to map the dimensionality-reduced frame vectors
    # back into the original space so that we can interpret them as frames
    noise_estimate = pca.inverse_transform(transformed_stack)
    noise_estimate = noise_estimate.reshape(stack.shape)

    return np.asarray(noise_estimate)


def get_pca_signal_estimates(
    stack: np.ndarray,
    parang: np.ndarray,
    pca_numbers: Iterable[int],
    roi_mask: Optional[np.ndarray] = None,
    return_components: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Get the signal estimate (i.e., the derotated and combined stack that
    has been denoised) using PCA-based PSF subtraction for different
    numbers of principal components.

    Note: This function essentially provides a rather minimalistic
    implementation of PynPoint's PcaPsfSubtractionModule.

    Args:
        stack: A 3D numpy array of shape `(n_frames, width, height)`
            containing the stack for which to estimate the systematic
            noise using PCA.
        parang: A numpy array of shape `(n_frames,)` which contains the
            respective parallactic angle for each frame (necessary for
            derotating the stack).
        pca_numbers: An iterable of integers, containing the values for
            the numbers of principal components for which to run PCA.
        roi_mask: A 2D binary mask of shape `(width, height)` that can
            be used to select the region of interest. If a ROI mask is
            given, only the pixels inside the ROI will be used to find
            the PCA basis.
        return_components: Whether or not to return the principal
            components of the PCA.

    Returns:
        A 3D numpy array of shape `(N, width, height)` (where N is the
        number of elements in `pca_numbers`), which contains the signal
        estimates for different numbers of principal components. The
        results are ordered from lowest to highest number of PCs.
    """

    # Convert pca_numbers into a sorted list
    pca_numbers = sorted(list(pca_numbers), reverse=True)

    # Find the maximum number of PCA components to use: This number cannot be
    # higher than the number of frames in the stack!
    max_pca_number = min(len(stack), max(pca_numbers))
    if max(pca_numbers) > len(stack):
        warn('n_components cannot be larger than n_frames!')

    # Reshape stack from 3D to 2D (each frame is turned into a single 1D
    # vector). If a ROI mask is given, only the pixels inside the ROI are
    # used; otherwise, all pixels are used.
    if roi_mask is not None:
        reshaped_stack = stack[:, roi_mask]
    else:
        reshaped_stack = stack.reshape(stack.shape[0], -1)

    # Instantiate new PCA with maximum number of principal components, and fit
    # it to the reshaped stack
    pca = PCA(n_components=max_pca_number)
    pca.fit(reshaped_stack)

    # Loop over different numbers of principal components and compute the
    # signal estimates for that number of PCs
    signal_estimates = []
    for n_components in pca_numbers:

        # Only keep the first `n_components` PCs
        truncated_pca = deepcopy(pca)
        truncated_pca.components_ = truncated_pca.components_[:n_components]

        # Apply the dimensionality-reducing transformation
        transformed_stack = truncated_pca.transform(reshaped_stack)

        # Use inverse transform to map the dimensionality-reduced frame vectors
        # back into the original space so that we can interpret them as frames
        noise_estimate_ = truncated_pca.inverse_transform(transformed_stack)
        if roi_mask is not None:
            noise_estimate = np.full(stack.shape, np.nan)
            noise_estimate[:, roi_mask] = noise_estimate_
        else:
            noise_estimate = noise_estimate_.reshape(stack.shape)

        # Compute the residual stack
        residual_stack = stack - noise_estimate

        # Derotate and combine the residuals to compute the signal estimate.
        # Do not use multiprocessing here, because nested multiprocessing is
        # probably a bad idea.
        signal_estimate = derotate_combine(
            stack=np.nan_to_num(residual_stack),
            parang=parang,
            n_processes=1,
        )

        # Restore ROI mask (if applicable)
        if roi_mask is not None:
            signal_estimate[~roi_mask] = np.nan

        signal_estimates.append((n_components, signal_estimate))

    # Sort the list such that signal estimates are ordered by increasing
    # number of principal components, and convert to a numpy array
    signal_estimates = sorted(signal_estimates, key=lambda _: int(_[0]))
    signal_estimates = np.array([_[1] for _ in signal_estimates])

    # Return the signal estimates and, if desired, also the principal
    # components reshaped to proper eigenimages / frames
    if not return_components:
        return np.asarray(signal_estimates)

    else:

        # The reshaping of principal components into 2D frames depends on
        # whether or not we have used an ROI mask
        if roi_mask is not None:
            components = np.full(
                (len(pca_numbers), stack.shape[1], stack.shape[2]), np.nan
            )
            components[:, roi_mask] = pca.components_
        else:
            components = deepcopy(pca.components_)
            components = components.reshape(
                (-1, stack.shape[1], stack.shape[2])
            )

        return signal_estimates, components
