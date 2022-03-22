"""
Methods for performing PCA-based PSF subtraction.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from copy import deepcopy
from typing import List, Optional, Tuple, Union, overload
from typing_extensions import Literal

from warnings import warn

from sklearn.decomposition import PCA

import numpy as np

from hsr4hci.derotating import derotate_combine
from hsr4hci.utils import check_consistent_size


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

@overload
def get_pca_signal_estimates(
    stack: np.ndarray,
    parang: np.ndarray,
    n_components: Union[int, List[int], Tuple[int], np.ndarray],
    return_components: Literal[True],
    roi_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    ...  # pragma: no cover


@overload
def get_pca_signal_estimates(
    stack: np.ndarray,
    parang: np.ndarray,
    n_components: Union[int, List[int], Tuple[int], np.ndarray],
    return_components: Literal[False],
    roi_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    ...  # pragma: no cover


def get_pca_signal_estimates(
    stack: np.ndarray,
    parang: np.ndarray,
    n_components: Union[int, List[int], Tuple[int], np.ndarray],
    return_components: bool = True,
    roi_mask: Optional[np.ndarray] = None,
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
        n_components: An iterable of integers, containing the values for
            the numbers of principal components for which to run PCA.
        return_components: Whether to return the principal components.
        roi_mask: A 2D binary mask of shape `(width, height)` that can
            be used to select the region of interest. If a ROI mask is
            given, only the pixels inside the ROI will be used to find
            the PCA basis.

    Returns:
        A 3D numpy array of shape `(N, width, height)` (where N is the
        number of elements in `pca_numbers`), which contains the signal
        estimates for different numbers of principal components. The
        results are ordered from lowest to highest number of PCs.
    """

    # -------------------------------------------------------------------------
    # Preparations
    # -------------------------------------------------------------------------

    # Check that stack and parallactic angles have matching sizes
    check_consistent_size(stack, parang, axis=0)

    # Define shortcuts
    n_frames, x_size, y_size = stack.shape
    frame_size = (x_size, y_size)

    # Turn a single integer into a list of integers
    if isinstance(n_components, int):
        n_components = [n_components]

    # Check
    condition_1 = isinstance(n_components, (list, tuple, np.ndarray))
    condition_2 = all(isinstance(_, int) for _ in n_components)
    if not condition_1 or not condition_2:
        raise ValueError('n_components must be a (sequence of) integer(s)!')

    # Convert the component numbers into a sorted list
    n_components = sorted(list(n_components), reverse=True)

    # Find the maximum number of PCA components to use: This number cannot be
    # higher than the number of frames in the stack!
    max_n_components = max(n_components)
    if max_n_components > len(stack):
        warn(
            UserWarning(
                '(Maximum of) n_components cannot be larger than n_frames! '
                'max(n_components) was assumed as n_frames.'
            )
        )
        max_n_components = len(stack)

    # -------------------------------------------------------------------------
    # Compute the PCA for the maximum number of components that we need
    # -------------------------------------------------------------------------

    # Reshape stack from 3D to 2D (each frame is turned into a single 1D
    # vector). If a ROI mask is given, only the pixels inside the ROI are
    # used; otherwise, all pixels are used.
    if roi_mask is not None:
        reshaped_stack = stack[:, roi_mask]
    else:
        reshaped_stack = stack[:, np.full(frame_size, True)]

    # Instantiate new PCA instance with maximum number of principal
    # components, and fit it to the reshaped stack
    original_pca = PCA(n_components=max_n_components)
    original_pca.fit(reshaped_stack)
    truncated_pca = deepcopy(original_pca)

    # -------------------------------------------------------------------------
    # Loop over the requested component numbers and compute signal estimates
    # -------------------------------------------------------------------------

    # Initialize an empty array for the signal estimates
    signal_estimates = np.full((len(n_components), x_size, y_size), np.nan)

    # Loop over different numbers of principal components and compute the
    # signal estimates for that number of PCs
    for i, n in enumerate(n_components):

        # Only keep the first n principal components
        truncated_pca.components_ = truncated_pca.components_[:n]

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

        # Subtract the noise estimate to compute the residual stack
        residual_stack = np.nan_to_num(stack - noise_estimate)

        # Derotate and combine the residuals to compute the signal estimate
        signal_estimate = derotate_combine(stack=residual_stack, parang=parang)

        # Restore ROI mask (if applicable)
        if roi_mask is not None:
            signal_estimate[~roi_mask] = np.nan

        # Store the signal estimate for the current number of PCs
        signal_estimates[len(n_components) - i - 1] = signal_estimate

    # -------------------------------------------------------------------------
    # Return the signal estimates and, optionally, also the PCs
    # -------------------------------------------------------------------------

    # Either return the signal estimates directly...
    if not return_components:
        return signal_estimates

    # ...or prepare the PCs first, and return them with the signal estimates
    else:

        # The reshaping of principal components into 2D frames depends on
        # whether we have used an ROI mask
        if roi_mask is not None:
            components = np.full((len(n_components), x_size, y_size), np.nan)
            for i in range(len(n_components)):
                components[i, roi_mask] = original_pca.components_[i]
        else:
            components = deepcopy(original_pca.components_)
            components = components.reshape((-1, x_size, y_size))

        return signal_estimates, components
