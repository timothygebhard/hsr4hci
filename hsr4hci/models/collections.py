"""
Provides PixelPredictorCollection classes.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from copy import deepcopy
from typing import Optional, Tuple

from sklearn.decomposition import PCA
from skimage.morphology import binary_dilation

import numpy as np

from hsr4hci.models.predictors import PixelPredictor

from hsr4hci.utils.adi_tools import derotate_frames
from hsr4hci.utils.forward_modeling import get_signal_stack, \
    get_collection_region_mask
from hsr4hci.utils.masking import get_positions_from_mask, get_circle_mask
from hsr4hci.utils.predictor_selection import get_default_grid_mask, \
    get_default_mask, get_santa_mask


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class PixelPredictorCollection:
    """
    Wrapper class around a collection of PixelPredictors.

    A collection consists of a collection region, which is given by the
    "sausage"-shaped trace of a planet in a forward model (or a single
    position, in case we are not using forward modeling), and a separate
    PixelPredictor instance for every position within this region.

    Quantities such as a detection map are then obtained by averaging
    the planet coefficient over all models in the the collection region.
    This is, in essence, a method to test if a suspected signal is
    consistent with the expected apparent motion that a real planet
    signal would exhibit in the data.

    Args:
        position: A tuple (x, y) containing the position for which
            to train a collection. Note: This corresponds to the
            position where the planet in the forward model will be
            placed at t=0, that is, in the first frame.
    """

    def __init__(self,
                 position: Tuple[int, int],
                 hsr_instance):

        # Store the constructor arguments
        self.m__position = position
        self.m__hsr_instance = hsr_instance

        # Initialize additional class variables
        self.m__collection_region = None
        self.m__predictors = dict()
        self.m__collection_name = \
            f'collection_{self.m__position[0]}_{self.m__position[1]}'

        # Get variables which can be inherited from parent
        self.m__use_forward_model = hsr_instance.m__use_forward_model
        self.m__config_model = hsr_instance.m__config_model
        self.m__config_sources = hsr_instance.m__config_sources

    def get_average_signal_coef(self) -> Optional[float]:
        """
        Compute the average signal coefficient for this collection.

        Returns:
            The average (by default: the median) planet coefficient of
            all models in this collection. In case the collection was
            trained without forward modeling, None is returned.
        """

        # We can only compute an average signal coefficient if we have
        # trained the collection using forward modeling
        if self.m__use_forward_model:

            # Collect signal coefficients for all predictors in the collection
            signal_coefs = [predictor.m__signal_coef for _, predictor
                            in self.m__predictors.items()]

            # Return the median of all signal coefficients in the collection
            return float(np.nanmedian(signal_coefs))

        # Otherwise, we just return None
        return None

    def preprocess_sources(self,
                           sources: np.ndarray,
                           planet_signal: Optional[np.ndarray]) -> np.ndarray:
        return sources

    def get_predictor_mask(self,
                           position):

        mask_type = self.m__config_sources['mask']['type']

        # Collect options for mask creation
        kwargs = {**dict(mask_size=self.m__hsr_instance.m__frame_size,
                         position=position,
                         lambda_over_d=self.m__hsr_instance.m__lambda_over_d,
                         pixscale=self.m__hsr_instance.m__pixscale),
                  **self.m__config_sources['mask']['parameters']}

        # If we are using the default mask, we are done here
        if mask_type == 'default':
            return get_default_mask(**kwargs)
        elif mask_type == 'default_grid':
            return get_default_grid_mask(**kwargs)
        elif mask_type == 'santa':
            return get_santa_mask(**kwargs)
        # use all pixel without the pixel covered by the collection
        elif mask_type == 'all':
            # estimate pixel covered by the collection
            collection_region = np.zeros(kwargs["mask_size"])
            positions = self.m__collection_region
            collection_region[tuple(zip(*positions))] = 1

            if "dilation_radius" in kwargs:
                dilation_radius = kwargs["dilation_radius"]
                # the mask size needs to be an odd number
                dilation_mask_size = int(np.round(dilation_radius * 2 + 2))
                dilation_mask_size += (dilation_mask_size-1) % 2

                collection_region = \
                    binary_dilation(collection_region,
                                    selem=get_circle_mask(
                                        (dilation_mask_size,
                                         dilation_mask_size),
                                        dilation_radius))

            # get all pixel covered by the HSR model
            hsr_region = self.m__hsr_instance.m__roi_mask

            return np.logical_and(np.logical_not(collection_region),
                                  hsr_region)
        else:
            # For unknown mask types, raise an error
            raise ValueError('Invalid choice for mask_type!')

    def precompute_pca(self,
                       stack: np.ndarray,
                       position: Tuple[int, int],
                       planet_signal: Optional[np.ndarray] = None
                       ) -> np.ndarray:
        """
        Precompute the PCA for a given position (i.e., a single pixel).

        Args:
            stack: A 3D numpy array of shape (n_frames, width, height)
                containing the stack of frames to train on.
            position: A tuple (x, y) containing the position for which
                to pre-compute the PCA.
            planet_signal:

        Returns:
            The `sources` for the given position.
        """

        # Define some shortcuts
        n_components = self.m__config_sources['pca_components']
        pca_mode = self.m__config_sources['pca_mode']
        sv_power = self.m__config_sources['sv_power']

        # Get predictor pixels ("sources", as opposed to "targets")
        predictor_mask = self.get_predictor_mask(position=position)

        sources = stack[:, predictor_mask]

        sources = self.preprocess_sources(sources=sources,
                                          planet_signal=planet_signal)

        # Set up the principal component analysis (PCA)
        pca = PCA(n_components=n_components)

        # Depending on the pca_mode, we either use the PCs directly...
        if pca_mode == 'temporal':

            # Fit the PCA to the data. We take the transpose of the sources
            # such that the  principal components found by the PCA are also
            # time series.
            pca.fit(X=sources.T)

            # Select the principal components, undo the transposition, and
            # multiply the them with the desired power of the singular values
            tmp_sources = pca.components_.T
            tmp_sources *= np.power(pca.singular_values_, sv_power)

        # ...or the original data projected onto the PCs
        elif pca_mode == 'spatial':

            # Fit the PCA, transform the data into the rotated coordinate
            # system, and then multiply with the desired power of the singular
            # values. This is equivalent to first multiplying the PCs with the
            # SVs and then projecting; however, fit_transform() is generally
            # more efficient.
            tmp_sources = pca.fit_transform(X=sources)
            tmp_sources *= np.power(pca.singular_values_, sv_power)

        else:
            raise ValueError('pca_mode must be one of the following: '
                             '"fit" or "fit_transform"!')

        return tmp_sources

    def train_collection(self,
                         stack: np.ndarray,
                         parang: np.ndarray,
                         psf_cropped: np.ndarray):
        """
        Train this collection.

        This function essentially contains a loop over all positions in
        the collection, for which a PixelPredictor is initialized and
        trained.

        Args:
            stack: A 3D numpy array of shape (n_frames, width, height)
                containing the training data.
            parang: A 1D numpy array of shape (n_frames,) containing the
                corresponding parallactic angles for the stack.
            psf_cropped: A 2D numpy containing the cropped and masked
                PSF template that will be used to compute the forward
                model.
        """

        # ---------------------------------------------------------------------
        # Get signal_stack and collection_region based on use_forward_model
        # ---------------------------------------------------------------------

        if self.m__use_forward_model:
            signal_stack = get_signal_stack(position=self.m__position,
                                            frame_size=stack.shape[1:],
                                            parang=parang,
                                            psf_cropped=psf_cropped)
            collection_region_mask = get_collection_region_mask(signal_stack)
            self.m__collection_region = \
                get_positions_from_mask(collection_region_mask)
        else:
            signal_stack = None
            self.m__collection_region = [self.m__position]

        # ---------------------------------------------------------------------
        # Loop over all positions in the collection region
        # ---------------------------------------------------------------------

        for position in self.m__collection_region:

            # -----------------------------------------------------------------
            # If necessary, pre-compute the sources for this position
            # -----------------------------------------------------------------

            # If the sources dictionary of the HSR instance does not contain
            # the current position, we need to pre-compute the PCA for it
            if position not in self.m__hsr_instance.m__sources.keys():
                sources = \
                    self.precompute_pca(stack=stack,
                                        position=position)
                self.m__hsr_instance.m__sources[position] = sources

            # Otherwise we can simply retrieve the sources from this dict
            else:
                sources = self.m__hsr_instance.m__sources[position]

            # -----------------------------------------------------------------
            # Collect the targets and, if needed, the forward model
            # -----------------------------------------------------------------

            # Get regression target
            targets = stack[:, position[0], position[1]]

            # Get planet signal (only if we are using a forward model)
            if self.m__use_forward_model:
                planet_signal = signal_stack[:, position[0], position[1]]
            else:
                planet_signal = None

            # -----------------------------------------------------------------
            # Create a new PixelPredictor, train it, and store it
            # -----------------------------------------------------------------

            # Create a new PixelPredictor instance for this position
            pixel_predictor = PixelPredictor(config_model=self.m__config_model)

            # Train pixel predictor for the selected sources and targets. The
            # augmentation of the sources with the planet_signal (in case it
            # is not None) happens automatically inside the PixelPredictor.
            pixel_predictor.train(sources=sources,
                                  targets=targets,
                                  planet_signal=planet_signal)

            # Add trained PixelPredictor to PixelPredictorCollection
            self.m__predictors[position] = pixel_predictor


class PlanetSafePixelPredictorCollection(PixelPredictorCollection):

    def __init__(self,
                 position: Tuple[int, int],
                 hsr_instance):

        super().__init__(position=position,
                         hsr_instance=hsr_instance)

        # Add additional class variables
        self.m__sources = dict()

    def train_collection(self,
                         stack: np.ndarray,
                         parang: np.ndarray,
                         psf_cropped: np.ndarray):

        # ---------------------------------------------------------------------
        # Construct signal stack
        # ---------------------------------------------------------------------

        signal_stack = get_signal_stack(position=self.m__position,
                                        frame_size=stack.shape[1:],
                                        parang=parang,
                                        psf_cropped=psf_cropped)

        collection_region_mask = get_collection_region_mask(signal_stack)
        self.m__collection_region = \
            get_positions_from_mask(collection_region_mask)

        # ---------------------------------------------------------------------
        # Loop over all positions in the collection region
        # ---------------------------------------------------------------------

        for position in self.m__collection_region:

            # -----------------------------------------------------------------
            # Collect targets, planet signal and sources for fit
            # -----------------------------------------------------------------

            # Get targets for regression
            targets = stack[:, position[0], position[1]]

            # Get planet signal for this position
            planet_signal = signal_stack[:, position[0], position[1]]

            # Compute sources for this position (and store them)
            sources = \
                self.precompute_pca(stack=stack,
                                    planet_signal=planet_signal,
                                    position=position)
            self.m__sources[position] = sources

            # -----------------------------------------------------------------
            # Create a new PixelPredictor, train it, and store it
            # -----------------------------------------------------------------

            # Create a new PixelPredictor instance for this position
            pixel_predictor = PixelPredictor(config_model=self.m__config_model)

            # Train pixel predictor for the selected sources and targets. The
            # augmentation of the sources with the planet_signal (in case it
            # is not None) happens automatically inside the PixelPredictor.
            pixel_predictor.train(sources=sources,
                                  targets=targets,
                                  planet_signal=planet_signal)

            # Add trained PixelPredictor to PixelPredictorCollection
            self.m__predictors[position] = pixel_predictor

    def preprocess_sources(self,
                           sources: np.ndarray,
                           planet_signal: Optional[np.ndarray]) -> np.ndarray:

        # ---------------------------------------------------------------------
        # Get sources and orthogonalize them w.r.t. the planet signal
        # ---------------------------------------------------------------------

        # Normalize planet_signal
        normalized_planet_signal = \
            planet_signal / np.linalg.norm(planet_signal)

        # Orthogonalize sources with respect to planet_signal
        sources_projected = \
            sources - np.outer(np.matmul(sources.T, normalized_planet_signal),
                               normalized_planet_signal).T

        # Make sure sources_projected is actually orthogonal to planet signal
        projection = np.matmul(sources_projected.T, planet_signal)
        assert np.allclose(projection, np.zeros_like(projection)), \
            'Orthogonalization failed!'

        return sources_projected

    def get_collection_residuals(self,
                                 stack: np.ndarray,
                                 parang: np.ndarray) -> np.ndarray:

        residuals = np.zeros_like(stack)

        # Loop over all predictors in collection
        for position, predictor in self.m__predictors.items():

            # Get sources for this position
            tmp_sources = self.m__sources[position]

            # Get model from predictor and remove signal component such that
            # the resulting model represents only the "noise part"
            noise_model = deepcopy(predictor.m__model)
            noise_model.coef_ = noise_model.coef_[:-1]

            # Use noise model to get noise prediction
            noise_prediction = noise_model.predict(tmp_sources)

            # Compute residuals for this position
            residuals[:, position[0], position[1]] = \
                stack[:, position[0], position[1]] - noise_prediction

        # Derotate residual stack (onto first frame)
        derotated_residuals = derotate_frames(stack=residuals,
                                              parang=(parang-parang[0]))

        # Select "cylinder"
        frame_size = self.m__hsr_instance.m__frame_size
        psf_radius = self.m__hsr_instance.m__config_psf_template['psf_radius']
        lambda_over_d = self.m__hsr_instance.m__lambda_over_d
        pixscale = self.m__hsr_instance.m__pixscale
        psf_radius_pixel = int(psf_radius * (lambda_over_d / pixscale))
        selection_mask = get_circle_mask(mask_size=frame_size,
                                         radius=psf_radius_pixel,
                                         center=self.m__position)

        derotated_residuals[:, ~selection_mask] = np.nan

        # Return result
        return np.nansum(derotated_residuals, axis=(1, 2))
