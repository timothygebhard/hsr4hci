"""
Provides HalfSiblingRegression class.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Optional, Tuple

from tqdm import tqdm

import numpy as np

from hsr4hci.models.collections import PixelPredictorCollection
from hsr4hci.utils.forward_modeling import crop_psf_template
from hsr4hci.utils.general import split_into_n_chunks
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
                 config: dict):

        # Store the constructor arguments
        self.m__config = config

        # Define useful shortcuts
        self.m__add_planet_column = config['experiment']['add_planet_column']
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
        self.m__cropped_psf_template = None

    def get_cropped_psf_template(self,
                                 psf_template: np.ndarray) -> np.ndarray:
        """
        Get a cropped version of the `psf_template`.

        Crop a given `psf_template` according to the options specified
        in the experiment configuration, or simply return the cropped
        PSF template if we have already computed the cropped version
        before.

        Args:
            psf_template: A 2D numpy array containing the raw,
                unsaturated PSF template.

        Returns:
            A 2D numpy array containing the cropped and masked PSF
            template (according to the experiment config).
        """

        # If we have not run this function before, we need to actually crop
        # the PSF template according to the options from the config file
        if self.m__cropped_psf_template is None:

            # Collect arguments for cropping
            crop_psf_template_arguments = \
                {'psf_template': psf_template,
                 'psf_radius': self.m__config_psf_template['psf_radius'],
                 'rescale_psf': self.m__config_psf_template['rescale_psf'],
                 'pixscale': self.m__pixscale,
                 'lambda_over_d': self.m__lambda_over_d}

            # Crop the PSF and store the result
            self.m__cropped_psf_template = \
                crop_psf_template(**crop_psf_template_arguments)

        # Return the cropped PSF template
        return self.m__cropped_psf_template

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
        coefficients = np.full(output_shape, np.nan)

        # Convert the dictionary of coefficients into an array
        for position, position_coefficients in tmp_coefficients.items():
            n_entries = len(position_coefficients)
            coefficients[:n_entries, position[0], position[1]] = \
                position_coefficients

        return coefficients

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
        detection_map = np.full(self.m__frame_size, np.nan)

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

    def get_residual_stack(self,
                           stack: np.ndarray) -> np.ndarray:

        # Initialize an empty residual stack
        residual_stack = np.full(stack.shape, np.nan)

        # collect residual lines in time
        for position, collection in \
                tqdm(self.m__collections.items(), ncols=80):
            residual_stack[:, position[0], position[1]] = \
                collection.get_collection_residuals(stack=stack)

        return residual_stack

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
                planet signal.
        """

        # Get positions of pixels in ROI
        roi_pixels = get_positions_from_mask(self.m__roi_mask)

        # Run training by looping over the ROI and calling train_position()
        for position in tqdm(roi_pixels, total=len(roi_pixels), ncols=80):
            self.train_position(position=position,
                                stack=stack,
                                parang=parang,
                                psf_template=psf_template)

    def train_position(self,
                       position: Tuple[int, int],
                       stack: np.ndarray,
                       parang: np.ndarray,
                       psf_template: np.ndarray):
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
            psf_template: A 2D numpy array containing the unsaturated
                PSF template which is used for forward modeling of the
                planet signal.
        """

        # Create a PixelPredictorCollection instance for this position
        collection = PixelPredictorCollection(position=position,
                                              hsr_instance=self)

        # Get the cropped PSF template
        psf_cropped = self.get_cropped_psf_template(psf_template=psf_template)

        # Train and save the collection for this position
        collection.train_collection(stack=stack,
                                    parang=parang,
                                    psf_cropped=psf_cropped)

        # Add to dictionary of trained collections
        self.m__collections[position] = collection

    def train_region(self,
                     region_idx: int,
                     n_regions: int,
                     stack: np.ndarray,
                     parang: np.ndarray,
                     psf_template: np.ndarray):
        """
        Train the models for a given region, defined by the region index
        and the total number of regions.

        This functions obtains the list of pixels in the region of
        interest, splits it into `n_regions` chunks, and then runs the
        training for all collections that correspond to pixels in the
        `region_idx`-th chunk.
        This allows to parallelize the training on the cluster using a
        fixed number of parallel jobs (namely `n_regions`), regardless
        of the number of pixels in the region of interest.

        Args:
            region_idx: The index of the region (i.e., chunk of the list
                of positions in the ROI) for which to run the training.
            n_regions: The total number of regions into which the ROI is
                split for training.
            stack: A 3D numpy array of shape (n_frames, width, height)
                containing the training data.
            parang: A 1D numpy array of shape (n_frames,) containing
                the corresponding parallactic angles for the stack.
            psf_template: A 2D numpy array containing the unsaturated
                PSF template which is used for forward modeling of the
                planet signal.
        """

        # Get positions of pixels in ROI as a list
        roi_pixels = get_positions_from_mask(self.m__roi_mask)

        # Split the list of ROI pixels into n_regions equal chunks, and select
        # the region for which to run this function
        all_regions = split_into_n_chunks(roi_pixels, n_regions)
        region = all_regions[region_idx]

        # Run training by looping over the region and calling train_position()
        for position in tqdm(region, total=len(region), ncols=80):
            self.train_position(position=position,
                                stack=stack,
                                parang=parang,
                                psf_template=psf_template)
