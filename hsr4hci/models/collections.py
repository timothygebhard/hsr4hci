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

import numpy as np

from hsr4hci.models.predictors import PixelPredictor

from hsr4hci.utils.forward_modeling import get_signal_stack, \
    get_collection_region_mask
from hsr4hci.utils.masking import get_positions_from_mask, get_circle_mask
from hsr4hci.utils.predictor_selection import get_default_grid_mask, \
    get_default_mask, get_santa_mask
from hsr4hci.utils.preprocess_sources import standardize_sources, \
    orthogonalize_sources, compute_pca_sources, randomly_project_sources

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
        self.m__sources = dict()

    def get_preprocessed_sources(self,
                                 stack: np.ndarray,
                                 planet_signal: Optional[np.ndarray],
                                 position: Tuple[int, int]) -> np.ndarray:
        """
        For a given `position`, select the predictor pixels from the
        given `stack` and apply a given series of preprocessing steps
        to them. Store the result so that we do not need to recompute
        it if the sources again.

        Args:
            stack: A 3D numpy array of shape (n_frames, width, height)
                containing the training data.
            planet_signal: A 1D numpy array of shape (n_frames,) which
                contains the planet signal from the forward model for
                the given `position`. This can be None, unless one of
                the pre-processing step explicitly requires it (e.g.,
                because we are orthogonalizing the sources w.r.t. the
                planet signal).
            position: A tuple (x, y) containing the position for which
                to select and pre-process the predictor pixels.

        Returns:
            A 2D numpy array of shape (n_frames, n_predictors), which
            contains the pre-processed sources for the given `position`.
        """

        # ---------------------------------------------------------------------
        # Check if we have computed these sources before already
        # ---------------------------------------------------------------------

        # If we have computed the sources for this position before already,
        # we do not need to recompute them, but return the cached result
        if position in self.m__sources.keys():
            return self.m__sources[position]

        # ---------------------------------------------------------------------
        # Get mask and select raw sources from stack
        # ---------------------------------------------------------------------

        # Get predictor mask according to specifications in config file
        predictor_mask = self.get_predictor_mask(position=position)

        # Use this mask to select the raw predictor pixels from the stack.
        # This will return a matrix of shape (n_frames, n_predictor_pixels).
        sources = stack[:, predictor_mask]

        # ---------------------------------------------------------------------
        # Apply various pre-processing steps to the raw sources
        # ---------------------------------------------------------------------

        for preprocessing_step in self.m__config_sources['preprocessing']:

            # Define shortcuts to pre-processing type and parameters
            preprocessing_type = preprocessing_step['type']
            parameters = preprocessing_step['parameters']

            # Standardize the features (columns) of the sources by subtracting
            # their mean and dividing by their standard deviation
            if preprocessing_type == 'standardize':
                sources = standardize_sources(sources=sources,
                                              parameters=parameters)

            # Orthogonalize the features (columns) of the sources w.r.t. a
            # particular vector (e.g., the planet signal) by removing the
            # projection onto that vector from every feature
            elif preprocessing_type == 'orthogonalize':
                sources = orthogonalize_sources(sources=sources,
                                                planet_signal=planet_signal,
                                                parameters=parameters)

            # Compute a PCA on the sources and then replace the sources by
            # either the PCs, or a projection of the data onto the PCs
            elif preprocessing_type == 'pca':
                sources = compute_pca_sources(sources=sources,
                                              parameters=parameters)

            # Reduce the dimensionality of the sources by using a random
            # projection (either Gaussian or sparse)
            elif preprocessing_type == 'random_projection':
                sources = randomly_project_sources(sources=sources,
                                                   parameters=parameters)

            # Raise an error if we do know the requested preprocessing type
            else:
                raise ValueError(f'Invalid preprocessing type encountered: '
                                 f'{preprocessing_type}')

        # ---------------------------------------------------------------------
        # Store pre-processed sources for later use and return them
        # ---------------------------------------------------------------------

        self.m__sources[position] = sources

        return sources

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
        mask_type = self.m__config_sources['selection_mask']['type']
        mask_params = self.m__config_sources['selection_mask']['parameters']
        frame_size = self.m__hsr_instance.m__frame_size
        lambda_over_d = self.m__hsr_instance.m__lambda_over_d
        pixscale = self.m__hsr_instance.m__pixscale

        # Collect all arguments in a dict
        kwargs = dict(mask_size=frame_size,
                      position=position,
                      lambda_over_d=lambda_over_d,
                      pixscale=pixscale,
                      **mask_params)

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
            sources = self.m__sources[position]

            # Get model from predictor and remove signal component such that
            # the resulting model represents only the "noise part"
            noise_model = deepcopy(predictor.m__model)
            if self.m__use_forward_model:
                noise_model.coef_ = noise_model.coef_[:-1]

            # Use noise model to get noise prediction
            noise_prediction = noise_model.predict(sources)

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

    @staticmethod
    def get_planet_signal(signal_stack: np.ndarray,
                          position: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Take the `signal_stack` at the given `position` and standardize
        (i.e., z-transform) the result before returning it. If the given
        `signal_stack` is None, return None.

        Args:
            signal_stack: A 3D numpy array of shape (n_frames, width,
                height) containing the forward model.
            position: A tuple (x, y) containing the spatial position for
                which to get the signal time series.

        Returns:
            The standardized signal_stack time series at the `position`.
        """

        # If the signal stack is None, we also return None
        if signal_stack is None:
            return None

        # Select planet signal
        planet_signal = signal_stack[:, position[0], position[1]]

        # Apply a z-transform to the planet signal
        planet_signal -= np.mean(planet_signal)
        planet_signal /= np.std(planet_signal, ddof=1)

        return planet_signal

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
            # Collect targets, sources and planet_signal for fit
            # -----------------------------------------------------------------

            # Get regression target
            targets = stack[:, position[0], position[1]]

            # Get planet signal (only if we are using a forward model)
            planet_signal = self.get_planet_signal(signal_stack=signal_stack,
                                                   position=position)

            # Get the pre-processed sources
            sources = \
                self.get_preprocessed_sources(stack=stack,
                                              planet_signal=planet_signal,
                                              position=position)

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
