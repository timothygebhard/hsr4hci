"""
Half-Sibling Regression model.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import joblib
import numpy as np
import os
import warnings

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
        self.m__pixscale = config['dataset']['pixscale']
        self.m__lambda_over_d = config['dataset']['lambda_over_d']
        self.m__roi_ier = config['experiment']['roi']['inner_exclusion_radius']
        self.m__roi_oer = config['experiment']['roi']['outer_exclusion_radius']
        self.m__config_collection = config['experiment']['collection']
        self.m__config_model = config['experiment']['model']
        self.m__config_psf_template = config['experiment']['psf_template']

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

    def get_detection_map(self):

        detection_map = np.full(self.m__frame_size, np.nan)
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

        # ---------------------------------------------------------------------
        # Pre-compute PCA
        # ---------------------------------------------------------------------

        # Run PCA for every position
        region_size = self.m__config_collection['predictor_region_radius']
        variance_threshold = \
            self.m__config_collection['explained_variance_threshold']

        # Compute the region for which we need to pre-compute the PCA
        psf_radius_pixel = np.ceil(self.m__config_psf_template['psf_radius'] *
                                   self.m__lambda_over_d / self.m__pixscale)
        roi_radius_pixel = np.ceil(self.m__roi_oer / self.m__pixscale)
        effective_radius = int(psf_radius_pixel + roi_radius_pixel + 2)
        pca_region_mask = get_circle_mask(mask_size=self.m__frame_size,
                                          radius=effective_radius)
        pca_region_positions = get_positions_from_mask(pca_region_mask)

        print("Pre-computing PCA:")
        for position in tqdm(pca_region_positions,
                             total=len(pca_region_positions), ncols=80):

            # Get predictor pixels
            predictor_mask = \
                get_predictor_mask(mask_size=tuple(stack.shape[1:]),
                                   position=position,
                                   region_size=region_size)
            sources = stack[:, predictor_mask]

            # Run PCA on sources and truncate based on a threshold criterion
            # on the explained variance of the principal components
            pca = PCA()
            sources = pca.fit_transform(X=sources)
            n_components = np.where(np.cumsum(pca.explained_variance_ratio_) >
                                    variance_threshold)[0][0] + 1
            self.m__sources[position] = sources[:, :n_components]

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

        # Run training
        print("\nTraining model for all positions in the ROI:")
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
        config_collection = self.m__config_collection
        collection = \
            PixelPredictorCollection(position=position,
                                     config_collection=config_collection,
                                     config_model=self.m__config_model)

        # Train and save the collection for this position
        collection.train_collection(stack=stack,
                                    parang=parang,
                                    sources=self.m__sources,
                                    psf_cropped=psf_cropped)

        # Add to dictionary of trained collections
        self.m__collections[position] = collection

    def predict(self,
                stack: np.ndarray):
        raise NotImplementedError()

    def load(self):

        # Get positions of pixels in ROI
        roi_pixels = get_positions_from_mask(self.m__roi_mask)

        # Load collection for every position in the ROI
        for position in roi_pixels:
            config_collection = self.m__config_collection
            collection = \
                PixelPredictorCollection(position=position,
                                         config_collection=config_collection,
                                         config_model=self.m__config_model)
            collection.load(models_root_dir=self.m__models_root_dir)
            self.m__collections[position] = collection

        # Restore pre-computed PCA sources
        file_path = os.path.join(self.m__models_root_dir, 'pca_sources.pkl')
        self.m__sources = joblib.load(filename=file_path)

    def save(self):

        # Save all PixelPredictorCollections
        for _, collection in self.m__collections.items():
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
                 config_collection: dict,
                 config_model: dict):

        self.m__position = position
        self.m__predictors = dict()
        self.m__config_model = config_model
        self.m__config_collection = config_collection
        self.m__collection_region = None
        
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

        # Find pixels with very large uncertainties and exclude them
        threshold = np.percentile(signal_sigma_coefs, 90)
        excluded_idx = np.where(signal_sigma_coefs > threshold)[0]
        signal_coefs = np.array(signal_coefs)[~excluded_idx]
        signal_sigma_coefs = np.array(signal_sigma_coefs)[~excluded_idx]

        # Return the weighted average of the signal coefficients
        return np.average(a=signal_coefs, weights=1/signal_sigma_coefs)

    def train_collection(self,
                         stack: np.ndarray,
                         parang: np.ndarray,
                         sources: dict,
                         psf_cropped: np.ndarray):

        # ---------------------------------------------------------------------
        # Get signal_stack and collection_region based on use_forward_model
        # ---------------------------------------------------------------------

        if self.m__config_collection['use_forward_model']:
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
                                           np.ones(stack.shape[0]),
                                           planet_signal.reshape(-1, 1)])

            # Create a new pixel predictor
            pixel_predictor = PixelPredictor(position=position,
                                             config_model=self.m__config_model)

            # Train pixel predictor
            pixel_predictor.train(sources=tmp_sources,
                                  targets=targets)

            # Add trained PixelPredictor to PixelPredictorCollection
            self.m__predictors[position] = pixel_predictor

    def save(self,
             models_root_dir: str):

        # Create folder for all models in this collection
        collection_dir = os.path.join(models_root_dir, self.m__collection_name)
        Path(collection_dir).mkdir(exist_ok=True)

        # Save file containing the positions of all pixels in the collection
        # TODO: Maybe use a FITS file instead pickling the data?
        file_path = os.path.join(collection_dir, 'positions.pkl')
        joblib.dump(self.m__collection_region, filename=file_path)

        # Loop over all PixelPredictors in the collection and save them
        for _, pixel_predictor in self.m__predictors.items():
            pixel_predictor.save(collection_dir=collection_dir)

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

        # Loop over all positions in the collection region and load
        # corresponding pixel predictors
        for position in self.m__collection_region:
            pixel_predictor = \
                PixelPredictor(position=position,
                               config_model=self.m__config_model)
            pixel_predictor.load(collection_dir=collection_dir)
            self.m__predictors[position] = pixel_predictor


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
        return self.coef_[-1], self.sigma_coef_[-1]

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

    def save(self,
             collection_dir: str):

        # Save the model itself
        file_path = os.path.join(collection_dir, self.m__name)
        joblib.dump(self.m__model, filename=file_path)

        # Save the coefficient uncertainties
        file_path = os.path.join(collection_dir, 'sigma__' + self.m__name)
        joblib.dump(self.sigma_coef_, filename=file_path)

    def load(self,
             collection_dir: str):

        # Try to load the model for this predictor from its *.pkl file
        file_path = os.path.join(collection_dir, self.m__name)
        if os.path.isfile(file_path):
            self.m__model = joblib.load(filename=file_path)
        else:
            warnings.warn(f'Model file not found: {file_path}')

        # Try to load the coefficient uncertainties for this predictor
        file_path = os.path.join(collection_dir, 'sigma__' + self.m__name)
        if os.path.isfile(file_path):
            self.sigma_coef_ = joblib.load(filename=file_path)
        else:
            warnings.warn(f'sigma_coef_ file not found: {file_path}')
