"""
Provides PixelPredictorCollection classes.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from copy import deepcopy
from typing import Optional, Tuple, TYPE_CHECKING

from photutils import CircularAperture, aperture_photometry
from skimage.morphology import binary_dilation
from sklearn.decomposition import PCA

import numpy as np

from hsr4hci.models.predictors import PixelPredictor

from hsr4hci.utils.forward_modeling import get_signal_stack, \
    get_collection_region_mask
from hsr4hci.utils.masking import get_positions_from_mask, get_circle_mask
from hsr4hci.utils.predictor_selection import get_default_grid_mask, \
    get_default_mask, get_santa_mask

# This is a somewhat ugly workaround to avoid circular dependencies when using
# the HalfSiblingRegression class for type hinting: TYPE_CHECKING is always
# False at runtime (thus avoiding circular imports), but mypy / PyCharm will
# be able to make use of the import statement to know what to expect from a
# variable that has been type-hinted as a HalfSiblingRegression instance.
# Source: https://stackoverflow.com/a/39757388/4100721
if TYPE_CHECKING:
    from hsr4hci.models.hsr import HalfSiblingRegression


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
                 hsr_instance: 'HalfSiblingRegression'):

        # Store the constructor arguments
        self.m__position = position
        self.m__hsr_instance = hsr_instance

        # Initialize additional class variables
        self.m__collection_region_mask = None
        self.m__collection_region_positions = None
        self.m__planet_positions = None
        self.m__predictors = dict()

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

    @staticmethod
    def preprocess_sources(sources: np.ndarray,
                           planet_signal: Optional[np.ndarray]) -> np.ndarray:
        """
        Apply pre-processing to `sources`. In particular: orthogonalize
        the sources with respect to the `planet_signal`.

        Args:
            sources: A 2D numpy array of shape (n_frames, n_predictors)
                that we want to pre-process.
            planet_signal: A 1D numpy array of shape (n_frames,)
                containing the planet signal from forward modeling.
                If this is None, no pre-processing is applied to the
                `sources`.

        Returns:
            A 2D numpy array of shape (n_frames, n_predictors) which
            contains the pre-processed (i.e., orthogonalized) `sources`.
        """

        # ---------------------------------------------------------------------
        # If we do not receive a planet_signal, we do not pre-process anything
        # ---------------------------------------------------------------------

        if planet_signal is None:
            return sources

        # ---------------------------------------------------------------------
        # Otherwise, orthogonalize the sources w.r.t. the planet_signal
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

    def get_predictor_mask(self,
                           position: Tuple[int, int]) -> np.ndarray:
        """
        Return the predictor mask for the given `position` within the
        collection region (using the collection region as the exclusion
        region for the mask).

        Args:
            position: A tuple (x, y) which specifies the position for
                which to compute the predictor selection mask.

        Returns:
            A 2D binary numpy array of shape (width, height) which masks
            the spatial pixels that should be used to make a prediction
            for the given `position`.
        """

        # ---------------------------------------------------------------------
        # Collect options for mask creation
        # ---------------------------------------------------------------------

        # Define some useful shortcuts
        mask_type = self.m__config_sources['mask']['type']
        frame_size = self.m__hsr_instance.m__frame_size
        lambda_over_d = self.m__hsr_instance.m__lambda_over_d
        pixscale = self.m__hsr_instance.m__pixscale

        # Collect all arguments in a dict
        kwargs = dict(mask_size=frame_size,
                      position=position,
                      lambda_over_d=lambda_over_d,
                      pixscale=pixscale,
                      **self.m__config_sources['mask']['parameters'])

        # ---------------------------------------------------------------------
        # Create collection-specific exclusion region
        # ---------------------------------------------------------------------

        # By default, the exclusion region is simply the collection region
        exclusion_mask = deepcopy(self.m__collection_region_mask)

        # If necessary, the exclusion region is dilated. This is useful in
        # cases where the radius of the PSF for the forward model is chosen
        # to be smaller than one lambda over D.
        if 'dilation_radius' in kwargs:

            # Select the dilation radius. This is the radius of the circular
            # mask that is used to dilate the exclusion mask, and it given in
            # units if lambda over D (because the cropping radius of the PSF
            # template is also specified in units of lambda over D).
            dilation_radius = kwargs['dilation_radius']

            # Convert dilation_radius from units of lambda over D to pixel
            dilation_radius_pixel = \
                int(dilation_radius * (lambda_over_d / pixscale))

            # Compute the size of the circular mask used for dilated the
            # exclusion region. This needs to be an odd number.
            dilation_mask_size = 2 * dilation_radius_pixel + 1

            # Create the circular mask for dilating the exclusion region
            dilation_mask = get_circle_mask(mask_size=(dilation_mask_size,
                                                       dilation_mask_size),
                                            radius=dilation_radius_pixel)

            # Dilate the default exclusion mask with the dilation mask
            exclusion_mask = binary_dilation(image=exclusion_mask,
                                             selem=dilation_mask)

        # ---------------------------------------------------------------------
        # Depending on the mask type, get the correct selection mask
        # ---------------------------------------------------------------------

        if mask_type == 'default':
            selection_mask = get_default_mask(**kwargs)
        elif mask_type == 'default_grid':
            selection_mask = get_default_grid_mask(**kwargs)
        elif mask_type == 'santa':
            selection_mask = get_santa_mask(**kwargs)
        elif mask_type == 'all':
            selection_mask = np.full(frame_size, True)
        else:
            raise ValueError('Invalid choice for mask_type!')

        # ---------------------------------------------------------------------
        # Remove the exclusion region from the selection mask and return result
        # ---------------------------------------------------------------------

        # Remove the exclusion_mask from the selection_mask
        predictor_mask = np.logical_and(selection_mask,
                                        np.logical_not(exclusion_mask))

        return predictor_mask

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

        # Apply pre-processing to sources: If planet_signal is not None, we
        # will project the sources into a subspace that is orthogonal to the
        # planet signal.
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

            # Compute the forward model, that is, the planet signal under the
            # assumption that the planet at t=0 is at self.m__position
            signal_stack, self.m__planet_positions = \
                get_signal_stack(position=self.m__position,
                                 frame_size=stack.shape[1:],
                                 parang=parang,
                                 psf_cropped=psf_cropped)

            # Compute the collection mask, that is, the mask of all pixels
            # which, under the above assumption, at some point in time contain
            # planet signal
            self.m__collection_region_mask = \
                get_collection_region_mask(signal_stack)

            # Turn the mask into a list of positions (x, y)
            self.m__collection_region_positions = \
                get_positions_from_mask(self.m__collection_region_mask)

        else:

            # Without forward modeling, there is no planet signal
            signal_stack = None

            # The collection region only consists of the current position.
            # We therefore create a collection region mask that only masks
            # the defining position of the collection.
            self.m__collection_region_mask = np.full(stack.shape[1:], False)
            self.m__collection_region_mask[self.m__position] = True

            # The list of positions in the collection only contains one element
            self.m__collection_region_positions = [self.m__position]

        # ---------------------------------------------------------------------
        # Loop over all positions in the collection region
        # ---------------------------------------------------------------------

        for position in self.m__collection_region_positions:

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
    """
    A "planet safe" version of the PixelPredictorCollection: The main
    difference is that in this version, the preprocess_sources() method
    computes a projection of the sources that is orthogonal to the
    planet signal. This means this class can only be used with forward
    modeling enabled. It also implements the get_collection_residuals()
    method (which does not make sense without forward modeling).
    """

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

        # Compute the forward model, that is, the planet signal under the
        # assumption that the planet at t=0 is at self.m__position
        signal_stack, self.m__planet_positions = \
            get_signal_stack(position=self.m__position,
                             frame_size=stack.shape[1:],
                             parang=parang,
                             psf_cropped=psf_cropped)

        # Compute the collection mask, that is, the mask of all pixels
        # which, under the above assumption, at some point in time contain
        # planet signal
        self.m__collection_region_mask = \
            get_collection_region_mask(signal_stack)

        # Turn the mask into a list of positions (x, y)
        self.m__collection_region_positions = \
            get_positions_from_mask(self.m__collection_region_mask)

        # ---------------------------------------------------------------------
        # Loop over all positions in the collection region
        # ---------------------------------------------------------------------

        for position in self.m__collection_region_positions:

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

    def get_collection_residuals(self,
                                 stack: np.ndarray) -> np.ndarray:
        """
        Compute residual time series for this collection.

        The idea here is the following: for every pixel in the
        collection, we get the prediction from the noise part of the
        learned models and subtract that from the `stack`, giving us
        a stack of residual time series (for spatial pixels within the
        collection region).
        We then go over every frame in the residual stack and place a
        circular mask (with a radius that matches the one of the cropped
        PSF template associated with the collection) at the expected
        position of the planet (based on the forward model associated
        with the collection). We sum up the residual flux in this
        aperture and use it as the residual for the corresponding time
        step. This means that in the end, we return a time series for
        the pixel at self.m__position, providing a single residual value
        for every frame.

        Args:
            stack: A 3D numpy array of shape (n_frames, width, height)
                containing the stack of frames to train on.

        Returns:
            A 1D numpy array of shape (n_frames,) containing the
            residual time series which was computed as described above.
        """

        # ---------------------------------------------------------------------
        # Define some useful shortcuts
        # ---------------------------------------------------------------------

        psf_radius = self.m__hsr_instance.m__config_psf_template['psf_radius']
        lambda_over_d = self.m__hsr_instance.m__lambda_over_d
        pixscale = self.m__hsr_instance.m__pixscale
        n_frames = stack.shape[0]

        # Convert the radius of the PSF from units of lambda over D to pixel
        psf_radius_pixel = psf_radius * (lambda_over_d / pixscale)

        # ---------------------------------------------------------------------
        # Get noise model predictions and compute residuals
        # ---------------------------------------------------------------------

        # Initialize an array for the residuals we will compute
        residuals = np.full(stack.shape, np.nan)

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

            # Compute residuals for this position by subtracting the noise
            # model prediction from the original stack (and remove the median)
            residual = stack[:, position[0], position[1]] - noise_prediction
            residual -= np.median(residual)

            # Store the median-removed residuals for this position
            residuals[:, position[0], position[1]] = residual

        # ---------------------------------------------------------------------
        # Measure and sum up residual flux at expected planet positions
        # ---------------------------------------------------------------------

        # Loop over frames, place a circular aperture mask with radius psf_size
        # at the expected position of the planet (according to the forward
        # model of the collection), and sum up the values selected by the mask.
        # This is similar to derotating the frames, but it should introduce
        # less interpolation artifacts (and furthermore keep a possible planet
        # PSF aligned).
        result = list()
        for i in range(n_frames):

            # Get the expected planet position from the forward model
            # NOTE: We need to flip the (x, y) coordinates because photutils
            #       uses a different convention for the coordinate system...
            expected_planet_position = self.m__planet_positions[i][::-1]

            # Place a circular aperture at the expected planet position (whose
            # radius matches the radius that the PSF template was cropped to)
            # on the current residual frame and sum up the values inside of it
            aperture = CircularAperture(positions=expected_planet_position,
                                        r=psf_radius_pixel)
            photometry = aperture_photometry(data=residuals[i],
                                             apertures=aperture)

            # Store the result (i.e., the aperture sum) for this frame
            result.append(float(photometry['aperture_sum']))

        return np.array(result)
