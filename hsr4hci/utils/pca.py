"""
Utility functions for performing principal component analysis (PCA).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from copy import deepcopy
from typing import Iterable, Tuple, Union

from joblib import delayed, Parallel
from sklearn.decomposition import PCA
from tqdm import tqdm

import numpy as np

from hsr4hci.utils.derotating import derotate_combine
from hsr4hci.utils.tqdm import tqdm_joblib


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

    return noise_estimate


def get_pca_signal_estimates(
    stack: np.ndarray,
    parang: np.ndarray,
    pca_numbers: Iterable[int],
    return_components: bool = True,
    n_processes: int = 4,
    verbose: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Get the signal estimate (i.e., the derotated and combined stack that
    has been denoised) using PCA-based PSF subtraction for different
    numbers of principal components.

    Note: This function essentially provides an extremely minimalistic
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
        return_components: Whether or not to return the principal
            components of the PCA.
        n_processes: Number of parallel processes to be used to process
            the different numbers of principal components. Choosing this
            value too high will actually decrease the performance (due
            to the increased process initialization costs)!
            If this value is chosen as 1, no multiprocessing is used and
            the different numbers of components are processed serially.
        verbose: Whether or not to print debugging information.

    Returns:
        A 3D numpy array of shape `(N, width, height)` (where N is the
        number of elements in `pca_numbers`), which contains the signal
        estimates for different numbers of principal components. The
        results are ordered from lowest to highest number of PCs.
    """

    def vprint(string: str, end: str = '\n') -> None:
        if verbose:
            print(string, end=end, flush=True)

    # Convert pca_numbers into a sorted list
    pca_numbers = sorted(list(pca_numbers), reverse=True)

    # Find the maximum number of PCA components to use: This number cannot be
    # higher than the number of frames in the stack!
    max_pca_number = min(len(stack), max(pca_numbers))

    # Reshape stack from 3D to 2D: each frame is turned into a single long
    # vector of length width * height
    reshaped_stack = stack.reshape(stack.shape[0], -1)

    # Instantiate new PCA with maximum number of principal components, and fit
    # it to the reshaped stack
    vprint('Fitting PCA with maximum number of components...', end=' ')
    pca = PCA(n_components=max_pca_number)
    pca.fit(reshaped_stack)
    vprint('Done!')

    # If desired, create an array with the principal components reshaped to
    # proper eigenimages / frames
    if return_components:
        components = deepcopy(pca.components_)
        components = components.reshape(-1, stack.shape[1], stack.shape[2])
    else:
        components = None

    # Define helper function to get signal estimate for a given n_components
    def get_signal_estimate(
        n_components: int, pca: PCA,
    ) -> Tuple[int, np.ndarray]:

        # Only keep the first `n_components` PCs
        truncated_pca = deepcopy(pca)
        truncated_pca.components_ = truncated_pca.components_[:n_components]

        # Apply the dimensionality-reducing transformation
        transformed_stack = truncated_pca.transform(reshaped_stack)

        # Use inverse transform to map the dimensionality-reduced frame vectors
        # back into the original space so that we can interpret them as frames
        noise_estimate = truncated_pca.inverse_transform(transformed_stack)
        noise_estimate = noise_estimate.reshape(stack.shape)

        # Compute the residual stack
        residual_stack = stack - noise_estimate

        # Derotate and combine the residuals to compute the signal estimate.
        # Do not use multiprocessing here, because nested multiprocessing is
        # probably a bad idea.
        signal_estimate = derotate_combine(
            stack=residual_stack, parang=parang, n_processes=1,
        )

        return n_components, signal_estimate

    # Use joblib to process the different values of n_components in parallel...
    if n_processes > 1:
        vprint('Computing signal estimates (in parallel):')
        with tqdm_joblib(tqdm(total=len(pca_numbers), ncols=80)) as _:
            with Parallel(n_jobs=n_processes, require='sharedmem') as run:
                signal_estimates = run(
                    delayed(get_signal_estimate)(n_components, pca)
                    for n_components in pca_numbers
                )

    # ...or simply serially if n_processes == 1
    else:
        vprint('Computing signal estimates (serially):')
        signal_estimates = list()
        for n_components in tqdm(pca_numbers, ncols=80):
            signal_estimates.append(get_signal_estimate(n_components, pca))

    # Sort the list such that signal estimates are ordered by increasing
    # number of principal components, and convert to a numpy array
    signal_estimates = sorted(signal_estimates, key=lambda _: _[0])
    signal_estimates = np.array([_[1] for _ in signal_estimates])

    # Return the signal estimates and optionally also the principal components
    if return_components:
        return signal_estimates, components
    return signal_estimates
