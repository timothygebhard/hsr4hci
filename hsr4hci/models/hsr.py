"""
Half-Sibling Regression model.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import joblib
import numpy as np
import os

from hsr4hci.models.prototypes import ModelPrototype
from hsr4hci.utils.forward_modeling import crop_psf_template, \
    get_signal_stack, get_collection_region_mask
from hsr4hci.utils.masking import get_circle_mask, get_positions_from_mask
from hsr4hci.utils.model_loading import get_class_by_name
from hsr4hci.utils.predictor_selection import get_predictor_mask
from hsr4hci.utils.roi_selection import get_roi_mask

from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm
from typing import Optional, Tuple


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

    def __init__(self,
                 config: dict):

        # Store the experiment configuration
        self.m__config_model = config['experiment']['model']
        self.m__config_psf_template = config['experiment']['psf_template']
        self.m__config_sources = config['experiment']['sources']
        self.m__lambda_over_d = config['dataset']['lambda_over_d']
        self.m__pixscale = config['dataset']['pixscale']
        self.m__roi_ier = config['experiment']['roi']['inner_exclusion_radius']
        self.m__roi_oer = config['experiment']['roi']['outer_exclusion_radius']
        self.m__use_forward_model = False

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

    def get_detection_map(self,
                          weighted=False):

        detection_map = np.full(self.m__frame_size, np.nan)
        for position, collection in self.m__collections.items():
            avg_weighted, avg = collection.get_average_signal_coef()
            if weighted:
                detection_map[position] = avg_weighted
            else:
                detection_map[position] = avg
        return detection_map

    def get_detection_stack(self):
        detection_frames = []
        for position, collection in self.m__collections.items():
            tmp_detection_frame = \
                collection.get_detection_frame(self.m__frame_size)
            tmp_detection_frame[position] = 0.0
            detection_frames.append(tmp_detection_frame)

        return np.array(detection_frames)

    def precompute_pca(self,
                       stack: np.ndarray):
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
        region_size = self.m__config_sources['predictor_region_radius']
        n_components = self.m__config_sources['n_pca_components']

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
        for position in tqdm(pca_region_positions,
                             total=len(pca_region_positions), ncols=80):

            # Get predictor pixels
            predictor_mask = \
                get_predictor_mask(mask_size=tuple(stack.shape[1:]),
                                   position=position,
                                   region_size=region_size)
            sources = stack[:, predictor_mask]

            # Set up a PCA and fit it to the predictor pixels
            # Note: We take the transpose of the sources, such that the
            # principal components found by the PCA are also time series
            # which we then use as the basis for fitting the noise.
            pca = PCA(n_components=n_components)
            pca.fit(X=sources.T)

            # Get the principal components and the mean (which is
            # automatically removed by the PCA) and stack them together
            pca_comp = pca.components_
            pca_mean = pca.mean_
            tmp_sources = np.row_stack([pca_comp,
                                        pca_mean.reshape((1, -1))])
            tmp_sources /= np.max(tmp_sources, axis=0)

            # Transpose back the result and and store it
            self.m__sources[position] = tmp_sources.T

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

        # If we have received a PSF template, we know we use forward modeling
        self.m__use_forward_model = psf_template is not None

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

        # Save all PixelPredictorCollections
        for _, collection in tqdm(self.m__collections.items(), ncols=80):
            collection.save(models_root_dir=self.m__models_root_dir)

        # Save pre-computed PCA sources
        file_path = os.path.join(self.m__models_root_dir, 'pca_sources.pkl')
        joblib.dump(self.m__sources, filename=file_path)


# -----------------------------------------------------------------------------


class PixelPredictorCollection(object):
    """
    Wrapper class ...
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

    def get_average_signal_coef(self):

        # Get signal coefficients and their uncertainties for all pixel
        # predictors in the collection
        signal_coefs = list()
        signal_sigma_coefs = list()
        for _, pixel_predictor in self.m__predictors.items():
            w_p, sigma_p = pixel_predictor.get_signal_coef()
            signal_coefs.append(w_p)
            signal_sigma_coefs.append(sigma_p)

        # 1.) Compute simple average
        average = np.median(signal_coefs)

        # 2.) Compute weighted average
        # Find pixels with very large uncertainties and exclude them
        threshold = np.percentile(signal_sigma_coefs, 90)
        excluded_idx = np.where(signal_sigma_coefs > threshold)[0]
        signal_coefs = np.array(signal_coefs)[~excluded_idx]
        signal_sigma_coefs = np.array(signal_sigma_coefs)[~excluded_idx]
        average_weighted = np.average(a=signal_coefs,
                                      weights=(1 / signal_sigma_coefs))

        return average_weighted, average

    def get_detection_frame(self,
                            frame_size):

        detection_frame = np.zeros(frame_size)
        for position, pixel_predictor in self.m__predictors.items():
            w_p, _ = pixel_predictor.get_signal_coef()
            detection_frame[position] = w_p

        return detection_frame

    def train_collection(self,
                         stack: np.ndarray,
                         parang: np.ndarray,
                         sources: dict,
                         psf_cropped: np.ndarray):

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

            # Get planet signal
            planet_signal = signal_stack[:, position[0], position[1]]

            # Get predictor pixels
            tmp_sources = sources[position]

            # Add planet signal from forward modeling to sources
            tmp_sources = np.column_stack([tmp_sources,
                                           planet_signal.reshape(-1, 1)])

            # Create a new pixel predictor
            pixel_predictor = PixelPredictor(position=position,
                                             config_model=self.m__config_model)

            # Train pixel predictor
            pixel_predictor.train(sources=tmp_sources,
                                  targets=targets)

            # Add trained PixelPredictor to PixelPredictorCollection
            self.m__predictors[position] = pixel_predictor

        # ---------------------------------------------------------------------
        # Clean up after training is complete (to use less memory)
        # ---------------------------------------------------------------------

        del signal_stack

    def save(self,
             models_root_dir: str):

        # Create folder for all models in this collection
        collection_dir = os.path.join(models_root_dir, self.m__collection_name)
        Path(collection_dir).mkdir(exist_ok=True)

        # Save file containing the positions of all pixels in the collection
        file_path = os.path.join(collection_dir, 'positions.pkl')
        joblib.dump(self.m__collection_region, filename=file_path)

        # Save the predictors
        file_path_predictors = os.path.join(collection_dir, 'predictors.pkl')
        joblib.dump(self.m__predictors, filename=file_path_predictors)

    def load(self,
             models_root_dir: str):

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


class PixelPredictor(object):
    """
    Wrapper class for a predictor model of a single pixel.
    """

    def __init__(self,
                 position: tuple,
                 config_model: dict):

        # Store constructor arguments
        self.m__position = position
        self.m__config_model = config_model

        # Create predictor name and placeholder for model
        self.m__name = f'model__{position[0]}_{position[1]}.pkl'
        self.m__model = None

        self.sigma_coef_ = None

    def get_signal_coef(self) -> Tuple[float, float]:
        return float(self.coef_[-1]), self.sigma_coef_[-1]

    @property
    def coef_(self) -> Optional[np.ndarray]:

        # If the base model has a coef_ attribute (which only exists for
        # fitted models), we can return it; otherwise return None
        if hasattr(self.m__model, 'coef_'):
            return self.m__model.coef_
        return None

    def train(self,
              sources: np.ndarray,
              targets: np.ndarray):

        # Instantiate a new model according to the model_config
        model_class = \
            get_class_by_name(module_name=self.m__config_model['module'],
                              class_name=self.m__config_model['class'])
        self.m__model = model_class(**self.m__config_model['parameters'])

        # Fit model to the training data
        self.m__model.fit(X=sources, y=targets)

        # Compute uncertainties for coefficients
        self.sigma_coef_ = np.diag(np.linalg.pinv(np.dot(sources.T, sources)))

    def predict(self,
                sources: np.ndarray) -> np.ndarray:

        return self.m__model.predict(X=sources)
