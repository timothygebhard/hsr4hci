"""
Provide a half-sibling regression (HSR) model.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Optional

import numpy as np

from hsr4hci.models.collections import PixelPredictorCollection
from hsr4hci.utils.model_loading import get_class_by_name


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class PixelPredictor:
    """
    Wrapper class for a predictor model of a single pixel.

    Args:
        collection_instance: Reference to the PixelPredictorCollection
            instance to which this PixelPredictor belongs.
    """

    def __init__(self, collection_instance: PixelPredictorCollection):

        # Store constructor arguments
        self.m__collection_instance = collection_instance

        # Initialize additional class variables
        self.m__model = None

        # Get variables which can be inherited from parents
        self.m__use_forward_model = collection_instance.m__use_forward_model
        self.m__config_model = collection_instance.m__config_model

    def get_signal_coef(self):
        """
        Get the coefficient for the planet signal for this predictor.

        Returns:
            The coefficient for the planet signal for this predictor.
        """

        if self.m__use_forward_model and self.m__model is not None:
            return self.m__model.coef_[-1]
        return None

    def train(self,
              sources: np.ndarray,
              targets: np.ndarray,
              planet_signal: Optional[np.ndarray] = None):
        """
        Train the model wrapper by the PixelPredictor.

        Args:
            sources: A 2D numpy array of shape (n_samples, n_features),
                which contains the training data (also known as the
                "independent variables") for the model.
            targets: A 1D numpy array of shape (n_samples,) that
                contains the regression targets (i.e, the "dependent
                variable") of the fit.
            planet_signal: A 1D numpy array containing the planet signal
                time series (from forward modeling) to be included in the
                model. May be None if `use_forward_model` is False.
        """

        # Instantiate a new model according to the model_config
        model_class = \
            get_class_by_name(module_name=self.m__config_model['module'],
                              class_name=self.m__config_model['class'])
        self.m__model = model_class(**self.m__config_model['parameters'])

        # Augment the sources: if we are using a forward model, we need to
        # add the planet signal as a new column to the sources here; if not,
        # we leave the sources unchanged
        if self.m__use_forward_model:

            # Sanity check: Make sure we actually got a planet signal
            if planet_signal is None:
                raise RuntimeError('use_forward_model is True, but no planet'
                                   'signal from forward modeling was provided')

            # Augment the sources by adding the planet signal as a new column
            sources = np.column_stack([sources, planet_signal.reshape(-1, 1)])

        # Fit model to the training data
        self.m__model.fit(X=sources, y=targets)

    def predict(self, sources: np.ndarray) -> np.ndarray:
        """
        Make predictions for given sources.

        Args:
            sources: A 2D numpy array of shape (n_samples, n_features),
                which contains the data for which we want to make a
                prediction using the trained model.

        Returns:
            A 1D numpy array of shape (n_samples, ) containing the
            model predictions for the given inputs (sources).
        """

        if self.m__model is not None:
            return self.m__model.predict(X=sources)
        raise RuntimeError('You called predict() on an untrained model!')
