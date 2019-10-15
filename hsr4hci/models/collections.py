"""
Provide a half-sibling regression (HSR) model.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Optional, Tuple

import numpy as np

from hsr4hci.models.hsr import HalfSiblingRegression
from hsr4hci.models.predictors import PixelPredictor
from hsr4hci.utils.forward_modeling import get_signal_stack, \
    get_collection_region_mask
from hsr4hci.utils.masking import get_positions_from_mask


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
                 hsr_instance: HalfSiblingRegression):

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
            signal_coefs = [predictor.get_signal_coef() for _, predictor
                            in self.m__predictors.items()]

            # Return the median of all signal coefficients in the collection
            return float(np.nanmedian(signal_coefs))

        # Otherwise, we just return None
        return None

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
                    self.m__hsr_instance.precompute_pca(stack=stack,
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
            pixel_predictor = PixelPredictor(collection_instance=self)

            # Train pixel predictor for the selected sources and targets. The
            # augmentation of the sources with the planet_signal (in case it
            # is not None) happens automatically inside the PixelPredictor.
            pixel_predictor.train(sources=sources,
                                  targets=targets,
                                  planet_signal=planet_signal)

            # Add trained PixelPredictor to PixelPredictorCollection
            self.m__predictors[position] = pixel_predictor
