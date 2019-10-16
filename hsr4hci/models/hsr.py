"""
Provides HalfSiblingRegression class.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from copy import deepcopy
from typing import Optional, Tuple, Type, Union

from tqdm import tqdm

import numpy as np

from hsr4hci.models.collections import PixelPredictorCollection as PPCollection
from hsr4hci.utils.forward_modeling import crop_psf_template, get_signal_stack
from hsr4hci.utils.masking import get_positions_from_mask
from hsr4hci.utils.roi_selection import get_roi_mask


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class HalfSiblingRegression:
    """
    Wrapper class for a half-sibling regression model.

    This class essentially encapsulates the "outer loop", that is,
    looping over every pixel in the (spatial) region of interest and
    learning a model (or a collection of models) for it.

    Args:
        config: A dictionary containing the experiment configuration.
    """

    def __init__(self,
                 config: dict,
                 collection_class: Type[PPCollection] = PPCollection):

        # Store the constructor arguments
        self.m__collection_class = collection_class
        self.m__config = config

        # Define useful shortcuts
        self.m__config_model = config['experiment']['model']
        self.m__config_psf_template = config['experiment']['psf_template']
        self.m__config_sources = config['experiment']['sources']
        self.m__experiment_dir = config['experiment_dir']
        self.m__frame_size = config['dataset']['frame_size']
        self.m__lambda_over_d = config['dataset']['lambda_over_d']
        self.m__pixscale = config['dataset']['pixscale']
        self.m__use_forward_model = config['experiment']['use_forward_model']

        # Compute implicitly defined class variables
        roi_ier = config['experiment']['roi']['inner_exclusion_radius']
        roi_oer = config['experiment']['roi']['outer_exclusion_radius']
        self.m__roi_mask = get_roi_mask(mask_size=self.m__frame_size,
                                        pixscale=self.m__pixscale,
                                        inner_exclusion_radius=roi_ier,
                                        outer_exclusion_radius=roi_oer)

        # Initialize additional class variables
        self.m__collections = dict()
        self.m__sources = dict()

    def get_cropped_psf_template(self,
                                 psf_template: np.ndarray) -> np.ndarray:
        """
        Crop a given `psf_template` according to the options specified
        in the experiment configuration.

        Args:
            psf_template: A 2D numpy array containing the raw,
                unsaturated PSF template.

        Returns:
            A 2D numpy array containing the cropped and masked PSF
            template (according to the experiment config).
        """

        # Crop and return the PSF template
        crop_psf_template_arguments = \
            {'psf_template': psf_template,
             'psf_radius': self.m__config_psf_template['psf_radius'],
             'rescale_psf': self.m__config_psf_template['rescale_psf'],
             'pixscale': self.m__pixscale,
             'lambda_over_d': self.m__lambda_over_d}
        return crop_psf_template(**crop_psf_template_arguments)

    def get_coefficients(self) -> np.ndarray:
        """
        Get all planet coefficients for all spatial positions.

        Returns:
            A numpy arrays containing the planet coefficients. The array
            has shape (max_size, width, height), where width and height
            refer to the spatial size of the stack on which the model
            was trained, and max_size is the number of pixels / models
            in the largest collection region.
            For all positions where the respective collection contains
            less pixels than max_size, the remaining array entries are
            filled with NaN; the same holds for all positions for which
            no collection was trained in the first place.
            This array may be useful to experiment with the way the
            detection map is computed. For example, a straightforward
            and simple way to obtain a detection map is to take the
            nanmedian along the first axis.
        """

        # Initialize dictionary to temporary hold the coefficients we collect
        tmp_coefficients = dict()

        # Keep track of the largest number of coefficients in a collection
        # (the size of a collection depends on its position)
        max_size = 0

        # Loop over all collections
        for position, collection in self.m__collections.items():

            # Loop over collection and collect the planet coefficients
            collection_coefficients = list()
            for _, predictor in collection.m__predictors.items():
                coefficient = predictor.get_signal_coef()
                collection_coefficients.append(coefficient)

            # Store them and update the maximum number of coefficients
            tmp_coefficients[position] = collection_coefficients
            max_size = max(max_size, len(collection_coefficients))

        # Define the shape of the output array and initialize it with NaNs
        output_shape = (max_size, ) + tuple(self.m__frame_size)
        coefficients = np.full(output_shape, np.nan).astype(np.float32)

        # Convert the dictionary of coefficients into an array
        for position, position_coefficients in tmp_coefficients.items():
            n_entries = len(position_coefficients)
            coefficients[:n_entries, position[0], position[1]] = \
                position_coefficients

        return coefficients

    def get_noise_predictions(self,
                              stack_or_shape: Union[np.ndarray, tuple]
                              ) -> np.ndarray:
        """
        Get the predictions of the noise part of the models we learned.

        Args:
            stack_or_shape: Either a 3D numpy array of shape (n_frames,
                width, height) containing a stack of frames on which
                the trained models should be evaluated, or just a tuple
                (n_frames, width, height) containing the shape of the
                original stack on which the data was trained. In the
                first case, we will compute the PCA on the new stack
                and apply the trained models on them to obtain a stack
                of predictions. In the second case, we only return the
                predictions of the models on the data that they were
                trained on (in which case we do not need to compute the
                PCA for the model inputs again).

        Returns:
            A 3D numpy array with the same shape as `stack_or_shape`
            that contains, at each position (x, y) in the region of
            interest, the prediction of the model for (x, y). The model
            to make the prediction is taken from the collection at the
            same position. For positions for which no model was trained,
            the prediction default to NaN (i.e., you might want to use
            np.nan_to_num() before subtracting the predictions from
            your data to get the residuals of the model).
        """

        # Define stack shape based on whether we have received a stack or
        # only the shape of the stack
        if isinstance(stack_or_shape, tuple):
            stack_shape = stack_or_shape
        else:
            stack_shape = stack_or_shape.shape

        # Initialize an array that will hold our predictions
        predictions = np.full(stack_shape, np.nan).astype(np.float32)

        # Loop over all positions in the ROI and the respective collections
        for position, collection in \
                tqdm(self.m__collections.items(), ncols=80):

            # Get a copy of the predictor for this position
            predictor = deepcopy(collection.m__predictors[position].m__model)

            # If we have trained the model with forward modeling, drop the
            # signal part of the model (we're only predicting the noise here)
            if self.m__use_forward_model:
                predictor.coef_ = predictor.coef_[:-1]

            # If necessary, pre-compute PCA on stack to build sources
            if isinstance(stack_or_shape, tuple):
                sources = self.m__sources[position]
            else:
                sources = collection.precompute_pca(stack=stack_or_shape,
                                                    position=position)

            # Make prediction for position and store in predictions array
            predictions[:, position[0], position[1]] = \
                predictor.predict(X=sources)

        return predictions

    def get_best_fit_planet_model(self,
                                  detection_map: np.ndarray,
                                  stack_shape: Tuple[int, int, int],
                                  parang: np.ndarray,
                                  psf_template: np.ndarray) -> np.ndarray:
        """
        Get the best fit planet model (BFPM).

        Args:
            detection_map: A 2D numpy array containing the detection map
                obtained with get_detection_map().
            stack_shape: A tuple containing the shape of the stack on
                which we trained the model, which will also be the shape
                of the array containing the best fit planet model.
            parang: A 1D numpy array containing the parallactic angles.
            psf_template: A 2D numpy array containing the unsaturated
                PSF template (raw, i.e., not cropped or masked).

        Returns:
            A 3D numpy array with the same shape as `stack` that
            contains our best fit planet model.
        """

        # Crop and mask the PSF template
        psf_cropped = self.get_cropped_psf_template(psf_template=psf_template)

        # Initialize the best fit planet model
        best_fit_planet_model = np.zeros(stack_shape).astype(np.float32)

        # Get positions where detection map is positive (we ignore negative
        # entries because they are do not make sense astrophysically)
        positive_pixels = \
            get_positions_from_mask(np.nan_to_num(detection_map) > 0)

        # Loop over all these positions
        for position in tqdm(positive_pixels, ncols=80):

            # Compute the weight according to the detection map
            factor = detection_map[position[0], position[1]]

            # Compute the forward model for this position
            signal_stack = get_signal_stack(position=position,
                                            frame_size=self.m__frame_size,
                                            parang=parang,
                                            psf_cropped=psf_cropped)

            # Add the forward model for this position to the best fit model
            best_fit_planet_model += factor * signal_stack

        # "Normalize" the best-fit planet model
        best_fit_planet_model /= np.max(best_fit_planet_model)
        best_fit_planet_model *= np.nanmax(detection_map)

        return best_fit_planet_model

    def get_detection_map(self) -> np.ndarray:
        """
        Collect the detection map for the model.

        A detection map contains, at each position (x, y) within the
        region of interest, the average planet coefficient, where the
        average is taken over all models that belong to the collection
        (i.e., the "sausage-shaped" planet trace region) for (x, y).
        By default, the median is used to average the coefficients.
        In case we trained the model with use_forward_model=False, the
        detection map is necessarily empty (because the model does not
        contain a coefficient for the planet signal).

        Returns:
            A 2D numpy array containing the detection map for the model.
        """

        # Initialize an empty detection map
        detection_map = np.full(self.m__frame_size, np.nan).astype(np.float32)

        # If we are not using a forward model, we obviously cannot compute a
        # detection map, hence we return an empty detection map
        if not self.m__use_forward_model:
            print('\nWARNING: You called get_detection_map() with '
                  'use_forward_model=False! Returned an empty detection map.')
            return detection_map

        # Otherwise, we can loop over all collections and collect the
        # coefficients corresponding to the planet part of the model
        for position, collection in self.m__collections.items():
            detection_map[position] = collection.get_average_signal_coef()

        return detection_map

    def train(self,
              stack: np.ndarray,
              parang: Optional[np.ndarray],
              psf_template: Optional[np.ndarray]):
        """
        Train the complete HSR model.

        This function is essentially only a loop over all functions in
        the region of interest; the actual training at each position
        happens in train_position().

        Args:
            stack: A 3D numpy array of shape (n_frames, width, height)
                containing the stack of frames to train on.
            parang: A numpy array of length n_frames containing the
                parallactic angle for each frame in the stack.
            psf_template: A 2D numpy array containing the unsaturated
                PSF template which is used for forward modeling of the
                planet signal. If None is given instead, no forward
                modeling is performed.
        """

        # Crop the PSF template to the size specified in the config
        psf_cropped = self.get_cropped_psf_template(psf_template=psf_template)

        # Get positions of pixels in ROI
        roi_pixels = get_positions_from_mask(self.m__roi_mask)

        # Run training by looping over the ROI and calling train_position()
        for position in tqdm(roi_pixels, total=len(roi_pixels), ncols=80):
            self.train_position(position=position,
                                stack=stack,
                                parang=parang,
                                psf_cropped=psf_cropped)

    def train_position(self,
                       position: Tuple[int, int],
                       stack: np.ndarray,
                       parang: np.ndarray,
                       psf_cropped: np.ndarray):
        """
        Train the models for a given `position`.

        Essentially, this function sets up a PixelPredictorCollection
        and trains it. The motivation for separating this into its own
        function was to simplify parallelization of the training on a
        batch queue based cluster (where every position could be
        trained independently in a separate job).

        Args:
            position: A tuple (x, y) containing the position for which
                to train a collection. Note: This corresponds to the
                position where the planet in the forward model will be
                placed at t=0, that is, in the first frame.
            stack: A 3D numpy array of shape (n_frames, width, height)
                containing the training data.
            parang: A 1D numpy array of shape (n_frames,) containing
                the corresponding parallactic angles for the stack.
            psf_cropped: A 2D numpy containing the cropped and masked
                PSF template that will be used to compute the forward
                model.
        """

        # Create a PixelPredictorCollection instance for this position
        collection = self.m__collection_class(position=position,
                                              hsr_instance=self)

        # Train and save the collection for this position
        collection.train_collection(stack=stack,
                                    parang=parang,
                                    psf_cropped=psf_cropped)

        # Add to dictionary of trained collections
        self.m__collections[position] = collection
