"""
Provide functions for pre-processing the predictor pixels.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Optional

from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.random_projection import SparseRandomProjection, \
    GaussianRandomProjection

import numpy as np


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def standardize_sources(sources: np.ndarray,
                        parameters: Optional[dict] = None) -> np.ndarray:
    """
    Standardize sources array by applying modified a z-transform.

    For each column in sources, we subtract the median of that column
    and scales the data according to the quantile range. This is more
    robust to outliers and should help to prevent some self-subtraction
    effects, because the median (unlike the mean) should not be affected
    by the presence of a planet signal in the data.

    Args:
        sources: A 2D numpy array of shape (n_frames, n_predictors)
            that we want to standardize.
        parameters: A dictionary containing the parameters of the
            standardization transform. Usually, this will be None.

    Returns:
        A 2D numpy array of shape (n_frames, n_predictors) that has
        been standardized in the way described above.
    """

    # Set default parameters
    if parameters is None:
        parameters = dict()

    # Set up RobustScaler using the given parameters
    robust_scaler = RobustScaler(**parameters)

    # Standardize the given sources using the RobustScaler
    sources_standardized = robust_scaler.fit_transform(X=sources)

    # Return the standardized sources
    return sources_standardized


def orthogonalize_sources(sources: np.ndarray,
                          planet_signal: Optional[np.ndarray],
                          parameters: dict) -> np.ndarray:
    """
    Orthogonalize the sources with respect to a given target.

    "Orthogonalize" means that each columns of the `sources` is
    projected onto a given target. Subsequently, this projection is
    subtracted from the respective column. This results in a sources
    array that is orthogonal to the target, meaning that the target
    cannot be written as a linear combination of the columns of the
    sources anymore.

    Args:
        sources: A 2D numpy array of shape (n_frames, n_predictors)
            that we want to orthogonalize.
        planet_signal: A 1D numpy array of shape (n_frames,)
            containing the planet signal from forward modeling.
            Only necessary if the "target" in `parameters` is
            "planet_signal"; otherwise, this may be None.
        parameters: A dictionary containing the parameters of the
            orthogonalization, such as, for example, the target
            (i.e., the vector w.r.t. which we orthogonalize).

    Returns:
        A 2D numpy array of shape (n_frames, n_predictors) which
        contains the orthogonalized `sources`.
    """

    # Define some shortcuts to the orthogonalization parameters
    target_str = parameters['target']

    # Define the target for the orthogonalization, that is, the vector with
    # respect to which we will orthogonalize the input sources
    if target_str == 'planet_signal':
        if planet_signal is None:
            raise ValueError('You requested to orthogonalize the sources'
                             'w.r.t. the planet_signal, but provided None'
                             'as the planet_signal.')
        target = planet_signal / np.linalg.norm(planet_signal)
    elif target_str == 'constant':
        constant = np.ones(len(sources))
        target = constant / np.linalg.norm(constant)
    else:
        raise ValueError(f'Invalid target for orthogonalization'
                         f'encountered: {target_str}')

    # Orthogonalize sources with respect to the target
    sources_orthogonalized = \
        sources - np.outer(np.matmul(sources.T, target), target).T

    # Make sure the result is actually orthogonal to the target
    projection = np.matmul(sources_orthogonalized.T, planet_signal)
    assert np.allclose(projection, np.zeros_like(projection)), \
        'Orthogonalization failed!'

    return sources_orthogonalized


def compute_pca_sources(sources: np.ndarray,
                        parameters: dict) -> np.ndarray:
    """
    Compute a PCA on `sources` (using the given `parameters`).

    Args:
        sources: A 2D numpy array of shape (n_frames, n_predictors) on
            which to compute the principal component analysis.
        parameters: A dictionary containing options for the PCA such as
            the number of components to use.

    Returns:
        A 2D numpy array containing the PCA-processed `sources`.
    """

    # Define some shortcuts to the PCA parameters
    n_components = parameters['n_components']
    pca_mode = parameters['pca_mode']
    sv_power = parameters['sv_power']

    # Set up the principal component analysis (PCA) with the given
    # number of principal components
    pca = PCA(n_components=n_components)

    # Depending on the pca_mode, we either use the PCs directly...
    if pca_mode == 'temporal':

        # Fit the PCA to the data. We take the transpose of the sources
        # such that the  principal components found by the PCA are also
        # time series.
        pca.fit(X=sources.T)

        # Select the principal components, undo the transposition, and multiply
        # the them with the desired power of the singular values
        pca_sources = pca.components_.T
        pca_sources *= np.power(pca.singular_values_, sv_power)

    # ...or the original data projected onto the PCs
    elif pca_mode == 'spatial':

        # Fit the PCA, transform the data into the rotated coordinate system,
        # and then multiply with the desired power of the singular values.
        # This is equivalent to first multiplying the PCs with the SVs and then
        # projecting; however, fit_transform() is generally more efficient.
        pca_sources = pca.fit_transform(X=sources)
        pca_sources *= np.power(pca.singular_values_, sv_power)

    else:
        raise ValueError('pca_mode must be one of the following: '
                         '"temporal" or "spatial"!')

    return pca_sources


def randomly_project_sources(sources: np.ndarray,
                             parameters: dict) -> np.ndarray:
    """
    Reduce the dimensionality of the sources by applying a random
    projection on the features.

    Essentially, this replaces all columns in sources by a fixed number
    of linear combinations of the columns. The coefficients of these
    linear combinations are chosen randomly.

    Args:
        sources: A 2D numpy array of shape (n_frames, n_predictors)
            on which to run the random projection.
        parameters: A dictionary containing options for the random
            projection, such as the "method" to use ("gaussian" or
            "sparse").

    Returns:
        A 2D numpy array containing the `sources` after applying a
        random projection to its features (columns).
    """

    # Define some shortcuts to the random projection parameters
    method = parameters['method']
    params = {k: v for k, v in parameters.items() if k != 'method'}

    # Set up a projector based on the method and its parameters
    if method == 'gaussian':
        projector = GaussianRandomProjection(**params)
    elif method == 'sparse':
        projector = SparseRandomProjection(**params)
    else:
        raise ValueError(f'Invalid method for random projection '
                         f'encountered: {method}')

    # Project the sources using the projector
    sources_projected = projector.fit_transform(X=sources)

    return sources_projected
