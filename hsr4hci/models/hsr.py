"""
Provide a half-sibling regression (HSR) model.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

import os

from scipy.ndimage import rotate
from sklearn.decomposition import PCA
from tqdm import tqdm

import joblib
import numpy as np

from hsr4hci.models.prototypes import ModelPrototype
from hsr4hci.utils.adi_tools import derotate_frames
from hsr4hci.utils.forward_modeling import crop_psf_template, \
    get_signal_stack, get_collection_region_mask
from hsr4hci.utils.masking import get_circle_mask, get_positions_from_mask
from hsr4hci.utils.model_loading import get_class_by_name
from hsr4hci.utils.predictor_selection import get_predictor_mask
from hsr4hci.utils.roi_selection import get_roi_mask


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class HalfSiblingRegression(ModelPrototype):
    """
    Wrapper class for a half-sibling regression model.

    This class essentially encapsulates the "outer loop", that is,
    looping over every pixel in the (spatial) region of interest and
    learning a model (or a collection of models) for it.

    Args:
        config: A dictionary containing the experiment configuration.
    """

    def __init__(self, config: dict):

        # Store the experiment configuration
        self.m__config_model = config['experiment']['model']
        self.m__config_psf_template = config['experiment']['psf_template']
        self.m__config_sources = config['experiment']['sources']
        self.m__lambda_over_d = config['dataset']['lambda_over_d']
        self.m__pixscale = config['dataset']['pixscale']
        self.m__roi_ier = config['experiment']['roi']['inner_exclusion_radius']
        self.m__roi_oer = config['experiment']['roi']['outer_exclusion_radius']
        self.m__use_forward_model = config['experiment']['use_forward_model']

        # Define shortcuts to config elements
        self.m__experiment_dir = config['experiment_dir']
        self.m__frame_size = config['dataset']['frame_size']

        # Get implicitly defined class variables
        self.m__roi_mask = get_roi_mask(mask_size=self.m__frame_size,
                                        pixscale=self.m__pixscale,
                                        inner_exclusion_radius=self.m__roi_ier,
                                        outer_exclusion_radius=self.m__roi_oer)

        # Define a models directory and ensure it exists
        self.m__models_root_dir = \
            os.path.join(self.m__experiment_dir, 'models')
        Path(self.m__models_root_dir).mkdir(exist_ok=True)

        # Initialize a dict that will hold all pixel models
        self.m__collections = dict()

        # Initialize a dict that will hold the PCA results for all positions
        self.m__sources = dict()

    def get_coefficients_and_uncertainties(self) -> Tuple[np.ndarray,
                                                          np.ndarray]:
        """
        Get the planet coefficients and uncertainties for all positions.

        Returns:
            Two numpy arrays, `coefficients` and `uncertainties`. Each
            has shape (max_size, width, height), where width and height
            refer to the spatial size of the stack on which the model
            was trained, and max_size is the number of pixels / models
            in the largest collection region.
            At each spatial position (x, y), the arrays hold all planet
            signal coefficients and their respective uncertainties. For
            all positions where the respective collection contains less
            pixels than max_size, the remaining array entries are filled
            with NaN; the same holds for all positions for which no
            collection was trained in the first place.
            These two arrays are useful to experiment with the way the
            detection map is computed. For example, a straightforward
            and simple way to obtain a detection map is to take the
            nanmedian along the first axis of the `coefficients` array.
        """

        # Initialize dictionaries to temporary hold the coefficients and
        # uncertainties that we will collect
        tmp_coefficients = dict()
        tmp_uncertainties = dict()

        # Keep track of the largest number of coefficients in a collection
        # (the size of a collection depends on its position)
        max_size = 0

        # Loop over all collections
        for position, collection in self.m__collections.items():

            # Collect the coefficients and uncertainties for this collection
            collection_coefficients = list()
            collection_uncertainties = list()
            for _, predictor in collection.m__predictors.items():
                coefficient, uncertainty = predictor.get_signal_coef()
                collection_coefficients.append(coefficient)
                collection_uncertainties.append(uncertainty)

            # Store them and update the maximum number of coefficients
            tmp_coefficients[position] = collection_coefficients
            tmp_uncertainties[position] = collection_uncertainties
            max_size = max(max_size, len(collection_coefficients))

        # Define the shape of the output arrays, and initialize them as numpy
        # arrays full of NaNs
        output_shape = (max_size, ) + tuple(self.m__frame_size)
        coefficients = np.full(output_shape, np.nan)
        uncertainties = np.full(output_shape, np.nan)

        # Convert the dictionary of coefficients into an array
        for position, position_coefficients in tmp_coefficients.items():
            n_entries = len(position_coefficients)
            coefficients[:n_entries, position[0], position[1]] = \
                position_coefficients

        # Convert the dictionary of uncertainties into an array
        for position, position_uncertainties in tmp_uncertainties.items():
            n_entries = len(position_uncertainties)
            uncertainties[:n_entries, position[0], position[1]] = \
                position_uncertainties

        return coefficients, uncertainties

    def get_predictions(self, stack_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Get the predictions of the models we have learned.

        Args:
            stack_shape: A tuple containing the shape of the stack on
                which we trained the model.

        Returns:
            A 3D numpy array with the same shape as `stack_shape` that
            contains at each position (x, y) in the region of interest
            the prediction of the model for (x, y) -- taken from the
            collection at the same position -- on the data that it was
            trained on (i.e., NOT on the given `stack`).
            For positions for which no model was trained, the prediction
            default to NaN (i.e., you might want to use np.nan_to_num()
            before subtracting the predictions from your data to get the
            residuals of the model).
        """

        predictions = np.full(stack_shape, np.nan)
        for position, collection in \
                tqdm(self.m__collections.items(), ncols=80):
            predictor = collection.m__predictors[position]
            predictions[:, position[0], position[1]] = \
                predictor.predict(self.m__sources[position])
        return predictions

    def get_best_fit_planet_model(self,
                                  detection_map: np.ndarray,
                                  stack_shape: Tuple[int, int, int],
                                  parang: np.ndarray,
                                  psf_template: np.ndarray) -> np.ndarray:
        """
        Get the best fit planet model (BFPM).

        TODO: Is this even the "right" way to compute the BFPM?

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

        # Crop the PSF template to the size specified in the config
        crop_psf_template_arguments = \
            {'psf_template': psf_template,
             'psf_radius': self.m__config_psf_template['psf_radius'],
             'rescale_psf': self.m__config_psf_template['rescale_psf'],
             'pixscale': self.m__pixscale,
             'lambda_over_d': self.m__lambda_over_d}
        psf_cropped = crop_psf_template(**crop_psf_template_arguments)

        # Initialize the best fit planet model
        best_fit_planet_model = np.zeros(stack_shape)

        # Loop over all pixels in the ROI
        roi_pixels = get_positions_from_mask(self.m__roi_mask)
        for position in tqdm(roi_pixels, total=len(roi_pixels), ncols=80):

            # Compute the forward model for this position
            signal_stack = get_signal_stack(position=position,
                                            frame_size=self.m__frame_size,
                                            parang=parang,
                                            psf_cropped=psf_cropped)

            # Compute the weight according to the detection map
            factor = detection_map[position[0], position[1]]

            # Add the forward model for this position to the best fit model
            best_fit_planet_model += factor * signal_stack

        # Normalize the result
        best_fit_planet_model /= np.nansum(detection_map)

        return best_fit_planet_model

    def get_difference_image(self,
                             stack: np.ndarray,
                             parang: np.ndarray) -> np.ndarray:
        """
        Compute a difference image, which is essentially obtained by
        subtracting the "noise model" from the real data. This image
        should then be in unit of flux and allow for a fair comparison
        with, e.g., PCA-based PSF subtraction.

        Args:
            stack: A 3D numpy array of shape (n_frames, width, height)
                containing the stack of frames which we trained the
                model on (and now want to subtract the noise model's
                predictions from).
            parang: A numpy array of shape (n_frames,) containing the
                parallactic angle for each frame in the stack.

        Returns:
            A 2D numpy array, containing the median (?) along the time
            axis of the difference between the original data and the
            noise model's predictions for any given pixel in the ROI.
        """

        # Compute rotation angles: Remove the offset (which usually makes sure
        # that "up == North" in the derotated frames). This is necessary here
        # because the position indices will lose their meaning if we orient
        # the frames to the North before computing the full difference image.
        rotation_angles = parang - parang[0]

        # Initialize the result
        difference_img = np.zeros(self.m__frame_size)

        # Initialize the residuals, in case are not using a forward model
        no_fm_residuals = None
        if not self.m__use_forward_model:
            no_fm_residuals = deepcopy(stack)

        # Loop over all pixels in the ROI
        for position_outer, collection in \
                tqdm(self.m__collections.items(), ncols=80):

            # Create an empty array that has the same size as the train stack
            # which will hold our predictions for every pixel and time step
            predictions = np.zeros(stack.shape)

            # Make predictions for the current collection
            for position_inner, predictor in collection.m__predictors.items():

                # Create the "systematics only" model: If the PixelPredictor
                # includes the forward model, we simply copy the model and set
                # the coefficient that corresponds to the planet model to 0
                if predictor.m__use_forward_model:
                    model = deepcopy(predictor.m__model)
                    model.coef_[-1] = 0
                else:
                    model = predictor.m__model

                # Create the sources for the prediction: We may still need to
                # add a dummy column for the "planet signal" because the model
                # expects that many input features. We simply set this column
                # to all zeroes because the corresponding model coefficient is
                # 0 anyway.
                if predictor.m__use_forward_model:
                    tmp_sources = self.m__sources[position_inner]
                    sources = predictor.augment_sources(tmp_sources)
                else:
                    sources = self.m__sources[position_inner]

                # Get the predictions of the model for this position
                model_preds = model.predict(X=sources)

                # Make predictions for the current position in the collection
                if not self.m__use_forward_model:
                    no_fm_residuals[:, position_outer[0],
                                    position_outer[1]] -= model_preds
                else:
                    predictions[:, position_inner[0], position_inner[1]] = \
                        model_preds

            # If we have used forward model, we need to derotate and average
            # here already (because we are averaging over the collection)
            if self.m__use_forward_model:

                # Subtract the noise model's predictions from the original data
                residuals = stack - predictions

                # Subtract the mean of the residuals
                residuals -= np.mean(residuals, axis=0)

                # If we have used a forward model, we need to derotate and
                # average the residuals over the collection
                derotated = derotate_frames(stack=residuals,
                                            parang=rotation_angles)
                averaged = np.median(derotated, axis=0)

                # Now, keep only the result for the current position_outer
                difference_img[position_outer[0], position_outer[1]] = \
                    averaged[position_outer[0], position_outer[1]]

        # In case we are not using a forward model, we can do the derotation
        # and averaging after the processing all collections
        if not self.m__use_forward_model:

            # Subtract the mean of the residuals
            no_fm_residuals -= np.mean(no_fm_residuals, axis=0)

            # Derotate the residuals
            derotated = derotate_frames(stack=no_fm_residuals,
                                        parang=rotation_angles)

            # Take the median along the time axis
            difference_img = np.median(derotated, axis=0)

        # After we have assembled the full difference image, we can derotate
        # it by the offset that is necessary to orient the result to the North
        # (which may be useful for comparisons with PCA)
        derotated_difference_image = rotate(input=difference_img,
                                            angle=-parang[0],
                                            reshape=False)

        # Finally, apply the ROI mask to set everything to NaN for which we
        # have not computed a result (to distinguish it from "zero residual")
        derotated_difference_image[~self.m__roi_mask] = np.nan

        return derotated_difference_image

    def get_detection_map(self, weighted=False) -> np.ndarray:
        """
        Collect the detection map for the model.

         A detection map contains, at each position (x, y) within the
         region of interest, the average planet coefficient, where the
         average is taken over all models that belong to the
         PixelPredictorCollection for (x, y). If `weighted` is True,
         then each coefficient is weighted with its inverse uncertainty.
         In case we trained the model with use_forward_model=False,
         the detection map is necessarily empty (because the model does
         not contain a coefficient for the planet signal).

        Args:
            weighted: Whether or not to return the average of the
                planet coefficient weighted by its respective inverse
                uncertainty.

        Returns:
            A 2D numpy array containing the detection map for the model.
        """

        # Initialize an empty detection map
        detection_map = np.full(self.m__frame_size, np.nan)

        # If we are not using a forward model, we obviously cannot compute a
        # detection map, hence we return an empty detection map
        if not self.m__use_forward_model:
            print('\nWARNING: You called get_detection_map() with '
                  'use_forward_model=False! Returned an empty detection map.')
            return detection_map

        # Otherwise, we can loop over all collections and collect the
        # coefficients corresponding to the planet part of the model
        for position, collection in \
                tqdm(self.m__collections.items(), ncols=80):

            # Get both the weighted and unweighted average and store it at
            # the correct position within the detection map
            avg_weighted, avg = collection.get_average_signal_coef()
            detection_map[position] = avg_weighted if weighted else avg

        return detection_map

    def precompute_pca(self, stack: np.ndarray):
        """
        Pre-compute the PCA of the predictor pixels for every position.

        Find all positions for which we at some point will need to learn
        a model, select the data for their respective predictor pixels,
        run PCA on them, and store the result.
        This serves the purpose of reducing computational redundancy:
        since many PixelPredictorCollection regions will overlap, we
        do not want to run the PCA computation repeatedly; instead, we
        only do it once up front.

        Args:
            stack: A 3D numpy array of shape (n_frames, width, height)
                containing the stack of frames to train on.
        """

        # Define some shortcuts
        n_components = self.m__config_sources['pca_components']
        pca_mode = self.m__config_sources['pca_mode']
        mask_type = self.m__config_sources['mask']['type']
        mask_params = self.m__config_sources['mask']['parameters']

        # Compute the region for which we need to pre-compute the PCA
        psf_radius_pixel = np.ceil(self.m__config_psf_template['psf_radius'] *
                                   self.m__lambda_over_d / self.m__pixscale)
        roi_radius_pixel = np.ceil(self.m__roi_oer / self.m__pixscale)
        effective_radius = int(psf_radius_pixel + roi_radius_pixel + 2)
        pca_region_mask = get_circle_mask(mask_size=self.m__frame_size,
                                          radius=effective_radius)
        pca_region_positions = get_positions_from_mask(pca_region_mask)

        # Loop over all pixels for which we need to select the predictor
        # pixels and run PCA on them
        for position in tqdm(pca_region_positions, ncols=80):

            # Collect options for mask creation
            mask_args = dict(mask_size=self.m__frame_size,
                             position=position,
                             mask_params=mask_params,
                             lambda_over_d=self.m__lambda_over_d,
                             pixscale=self.m__pixscale)

            # Get predictor pixels ("sources", as opposed to "targets")
            predictor_mask = get_predictor_mask(mask_type=mask_type,
                                                mask_args=mask_args)
            sources = stack[:, predictor_mask]

            # Set up the principal component analysis (PCA)
            pca = PCA(n_components=n_components)

            # Depending on the pca_mode, we either use the PCs directly...
            if pca_mode == 'fit':

                # Fit the PCA to the data
                # Note: We take the transpose of the sources, such that the
                # principal components found by the PCA are also time series.
                pca.fit(X=sources.T)

                # Get the principal components and the mean (which is
                # automatically removed by the PCA) and stack them together
                tmp_sources = np.row_stack([pca.components_,
                                            pca.mean_.reshape((1, -1))])

                # Normalize such that the maximum value of every time series
                # is 1, and undo the transpose again
                # TODO: Is this a good way of "normalizing" the sources?
                # tmp_sources /= np.max(tmp_sources, axis=0)
                tmp_sources = tmp_sources.T

            # ...or the original data projected onto the PCs
            elif pca_mode == 'fit_transform':

                # Fit the transform and project the data onto the PCs
                tmp_sources = pca.fit_transform(X=sources)

            else:
                raise ValueError('pca_mode must be one of the following: '
                                 '"fit" or "fit_transform"!')

            self.m__sources[position] = tmp_sources

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

        # ---------------------------------------------------------------------
        # Basic sanity checks and preliminaries
        # ---------------------------------------------------------------------

        # Make sure we have called self.precompute_pca() before start training
        if not self.m__sources:
            print('\nself.m__sources was empty! Running PCA pre-computation:')
            self.precompute_pca(stack)
            print()

        # ---------------------------------------------------------------------
        # Crop the PSF template to the size specified in the config
        # ---------------------------------------------------------------------

        # Crop the PSF template to the size specified in the config
        crop_psf_template_arguments = \
            {'psf_template': psf_template,
             'psf_radius': self.m__config_psf_template['psf_radius'],
             'rescale_psf': self.m__config_psf_template['rescale_psf'],
             'pixscale': self.m__pixscale,
             'lambda_over_d': self.m__lambda_over_d}
        psf_cropped = crop_psf_template(**crop_psf_template_arguments)

        # ---------------------------------------------------------------------
        # Train models for every position in the ROI
        # ---------------------------------------------------------------------

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
            parang: A 1D numpy array of shape (n_frames,) containing the
                corresponding parallactic angles for the stack.
            psf_cropped: A 2D numpy containing the cropped and masked
                PSF template that will be used to compute the forward
                model.
        """

        # Create a PixelPredictorCollection for this position
        use_forward_model = self.m__use_forward_model
        collection = \
            PixelPredictorCollection(position=position,
                                     config_model=self.m__config_model,
                                     use_forward_model=use_forward_model)

        # Train and save the collection for this position
        collection.train_collection(stack=stack,
                                    parang=parang,
                                    sources=self.m__sources,
                                    psf_cropped=psf_cropped)

        # Add to dictionary of trained collections
        self.m__collections[position] = collection

    def load(self):
        """
        Load this HSR instance and its associated data from disk.
        """

        # Get positions of pixels in ROI
        roi_pixels = get_positions_from_mask(self.m__roi_mask)

        # Load collection for every position in the ROI
        use_forward_model = self.m__use_forward_model
        for position in tqdm(roi_pixels, ncols=80):
            collection = \
                PixelPredictorCollection(position=position,
                                         config_model=self.m__config_model,
                                         use_forward_model=use_forward_model)
            collection.load(models_root_dir=self.m__models_root_dir)
            self.m__collections[position] = collection

        # Restore pre-computed PCA sources
        file_path = os.path.join(self.m__models_root_dir, 'pca_sources.pkl')
        self.m__sources = joblib.load(filename=file_path)

    def save(self):
        """
        Save this HSR instance and its associated data to disk.
        """

        # Save all PixelPredictorCollections
        for _, collection in tqdm(self.m__collections.items(), ncols=80):
            collection.save(models_root_dir=self.m__models_root_dir)

        # Save pre-computed PCA sources
        file_path = os.path.join(self.m__models_root_dir, 'pca_sources.pkl')
        joblib.dump(self.m__sources, filename=file_path)


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
        config_model: A dictionary containing the configuration for the
            model that is wrapped by the PixelPredictor, for example,
            a sklearn.linear_model.LinearRegression.
        use_forward_model: Whether or not we are using forward modeling.
    """

    def __init__(self,
                 position: Tuple[int, int],
                 config_model: dict,
                 use_forward_model: bool):

        self.m__collection_region = None
        self.m__config_model = config_model
        self.m__position = position
        self.m__predictors = dict()
        self.m__use_forward_model = use_forward_model

        self.m__collection_name = \
            f'collection_{self.m__position[0]}_{self.m__position[1]}'

    def get_average_signal_coef(self) -> Tuple[Optional[float],
                                               Optional[float]]:
        """
        Compute the average signal coefficient for this collection.

        Returns:
            A tuple (weighted_average, average) containing the weighted
            average (using the inverse coefficient uncertainties) and
            the median of the planet signal coefficients in the
            collection. In case the collection was trained without
            forward modeling, this method will return (None, None).
        """

        # We can only compute an average signal coefficient if we have
        # trained the collection using forward modeling
        if self.m__use_forward_model:

            # Get signal coefficients and their uncertainties for all pixel
            # predictors in the collection
            signal_coefs = list()
            signal_sigma_coefs = list()
            for _, pixel_predictor in self.m__predictors.items():
                w_p, sigma_p = pixel_predictor.get_signal_coef()
                signal_coefs.append(w_p)
                signal_sigma_coefs.append(sigma_p)

            # Compute the median of all planet coefficients in the collection
            average = np.median(signal_coefs)

            # Compute a weighted average. To make this more robust, we first
            # identify pixels with very large uncertainties and exclude them.
            # TODO: This can probably still be improved a lot!
            threshold = np.percentile(signal_sigma_coefs, 90)
            excluded_idx = np.where(signal_sigma_coefs > threshold)[0]
            signal_coefs = np.array(signal_coefs)[~excluded_idx]
            signal_sigma_coefs = np.array(signal_sigma_coefs)[~excluded_idx]
            average_weighted = np.average(a=signal_coefs,
                                          weights=(1 / signal_sigma_coefs))

            return float(average_weighted), float(average)

        # Otherwise, we just return None
        return None, None

    def get_detection_frame(self, frame_size: Tuple[int, int]) -> np.ndarray:
        """
        Construct a frame where every pixel that is contained in the
        collection contains its respective planet signal coefficient,
        and all other pixels are set to NaN.

        Args:
            frame_size: A tuple (width, height) containing the size of
                the detection frame to be created. This of course needs
                to match the spatial size of the stack that was used
                to train the collection.

        Returns:
            A "detection frame", containing the planet coefficient of
            every PixelPredictor model in the collection at its
            respective spatial position.
        """

        detection_frame = np.full(frame_size, np.nan)
        for position, pixel_predictor in self.m__predictors.items():
            w_p, _ = pixel_predictor.get_signal_coef()
            detection_frame[position] = w_p

        return detection_frame

    def train_collection(self,
                         stack: np.ndarray,
                         parang: np.ndarray,
                         sources: dict,
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
            sources: A 2D numpy array of shape (n_frames, n_features)
                containing the pre-computed sources for the fit.
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

            # Get regression target
            targets = stack[:, position[0], position[1]]

            # Get planet signal (only if we are using a forward model)
            if self.m__use_forward_model:
                planet_signal = signal_stack[:, position[0], position[1]]
            else:
                planet_signal = None

            # Get predictor pixels
            tmp_sources = sources[position]

            # Create a new pixel predictor
            pixel_predictor = \
                PixelPredictor(config_model=self.m__config_model,
                               planet_signal=planet_signal,
                               use_forward_model=self.m__use_forward_model)

            # Train pixel predictor
            pixel_predictor.train(sources=tmp_sources,
                                  targets=targets)

            # Add trained PixelPredictor to PixelPredictorCollection
            self.m__predictors[position] = pixel_predictor

        # Clean up after training is complete (to use less memory)
        del signal_stack

    def save(self, models_root_dir: str):
        """
        Save this collection and its associated data to pickle files.

        Args:
            models_root_dir: Path to the models root directory (which
                contains a subdirectory for every collection).
        """

        # Create folder for all models in this collection
        collection_dir = os.path.join(models_root_dir, self.m__collection_name)
        Path(collection_dir).mkdir(exist_ok=True)

        # Save file containing the positions of all pixels in the collection
        file_path = os.path.join(collection_dir, 'positions.pkl')
        joblib.dump(self.m__collection_region, filename=file_path)

        # Save the predictors
        file_path_predictors = os.path.join(collection_dir, 'predictors.pkl')
        joblib.dump(self.m__predictors, filename=file_path_predictors)

    def load(self, models_root_dir: str):
        """
        Load this collection and its associated data from pickle files.

        Args:
            models_root_dir: Path to the models root directory (which
                contains a subdirectory for every collection).
        """

        # Construct name of directory that contains this collection, and check
        # if it exists
        collection_dir = os.path.join(models_root_dir, self.m__collection_name)
        if not os.path.isdir(collection_dir):
            raise NotADirectoryError(f'{collection_dir} does not exist!')

        # Load positions that belong to this collection
        file_path = os.path.join(collection_dir, 'positions.pkl')
        self.m__collection_region = joblib.load(filename=file_path)

        # Load the predictors
        file_path_predictors = os.path.join(collection_dir, 'predictors.pkl')
        self.m__predictors = joblib.load(filename=file_path_predictors)

# -----------------------------------------------------------------------------


class PixelPredictor:
    """
    Wrapper class for a predictor model of a single pixel.

    Args:
        config_model: A dictionary containing the configuration for the
            model that is wrapped by the PixelPredictor, for example,
            a sklearn.linear_model.LinearRegression.
        planet_signal: A 1D numpy array containing the planet signal
            time series (from forward modeling) to be included in the
            model. May be None if `use_forward_model` is False.
        use_forward_model: Whether or not we are using forward modeling.
    """

    def __init__(self,
                 config_model: dict,
                 planet_signal: Optional[np.ndarray],
                 use_forward_model: bool):

        # Store constructor arguments
        self.m__config_model = config_model
        self.m__planet_signal = planet_signal
        self.m__use_forward_model = use_forward_model

        # Initialize variables that we need for the train() method
        self.m__model = None
        self.sigma_coef_ = None

    def get_signal_coef(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Get the model coefficient corresponding to the planet signal
        and its respective uncertainty.

        Returns:
            A tuple (coef, sigma_coef) with the coefficient and its
            uncertainty. If the model was trained without forward
            modeling, these values will be None.
        """

        # Of course, we can only return a planet coefficient if we have
        # trained the model using forward modeling
        if self.m__use_forward_model:
            return (float(self.m__model.coef_[-1]),
                    float(self.sigma_coef_[-1]))
        return None, None

    def augment_sources(self, sources) -> np.ndarray:
        """
        Augment the given `sources` by self.m__planet_signal by adding
        it as a column to `sources` and return the result.

        Args:
            sources: A 2D numpy array of shape (n_samples, n_features),
                containing the "independent variables" for the fit.

        Returns:
            The augmented sources, that is, a 2D numpy array of shape
            (n_samples, n_features + 1), where the last column is
            given by self.m__planet_signal.
        """

        # We only need to augment the sources if we are using forward
        # modeling, otherwise we can return them unaltered
        if self.m__use_forward_model:
            planet_signal = self.m__planet_signal.reshape(-1, 1)
            sources = np.column_stack([sources, planet_signal])
        return sources

    def train(self,
              sources: np.ndarray,
              targets: np.ndarray):
        """
        Train the model wrapper by the PixelPredictor.

        Args:
            sources: A 2D numpy array of shape (n_samples, n_features),
                which contains the training data (also known as the
                "independent variables") for the model.
            targets: A 1D numpy array of shape (n_samples,) that
                contains the regression targets (i.e, the "dependent
                variable") of the fit.
        """

        # Instantiate a new model according to the model_config
        model_class = \
            get_class_by_name(module_name=self.m__config_model['module'],
                              class_name=self.m__config_model['class'])
        self.m__model = model_class(**self.m__config_model['parameters'])

        # Augment the sources: If we are using a forward model, we need to
        # add the planet signal as a new column to the sources here; if not,
        # we leave the sources unchanged
        sources = self.augment_sources(sources)

        # Fit model to the training data
        self.m__model.fit(X=sources, y=targets)

        # Compute uncertainties for coefficients
        # TODO: This is probably only correct for vanilla linear regression?
        self.sigma_coef_ = np.diag(np.linalg.pinv(np.dot(sources.T, sources)))

    def predict(self, sources: np.ndarray) -> np.ndarray:
        """
        Use the trained model to make a prediction on the given input.

        Args:
            sources: A 2D numpy array of shape (n_samples, n_features),
                which contains the data for which we want to make a
                prediction using the model of this PixelPredictor.

        Returns:
            A 1D numpy array of shape (n_samples,) which contains the
            predictions of the model for the given `sources`.
        """

        # We can only make a prediction if we have trained the model already
        if self.m__model is not None:

            # Augment sources based on self.m__use_forward model
            sources = self.augment_sources(sources)

            # Make prediction and return it
            return self.m__model.predict(X=sources)

        raise RuntimeError('You tried to call predict() before actually '
                           'training the model!')
