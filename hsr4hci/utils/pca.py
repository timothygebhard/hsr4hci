"""
Utility functions for performing principal component analysis (PCA).
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from sklearn.decomposition import PCA

import numpy as np


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

    Note: This function essentially provides an extremely minimalistic
    implementation of PynPoint's PcaPsfSubtractionModule.

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
    noise_estimate = \
        pca.inverse_transform(transformed_stack).reshape(stack.shape)

    return noise_estimate
