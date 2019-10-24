"""
Provides PixelPredictor class.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Optional

import numpy as np

from hsr4hci.utils.model_loading import get_class_by_name


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class PixelPredictor:
    """
    Wrapper class for a predictor model of a single pixel.

    Args:
        config_model: A dictionary containing the configuration for
            the model that the predictor is based on (e.g., a RidgeCV
            regressor from sklearn.linear_model).
    """

    def __init__(self,
                 config_model: dict):

        # Initialize additional class variables
        self.m__model = None

        # Get variables which can be inherited from parents
        self.m__config_model = config_model

    @property
    def m__signal_coef(self) -> Optional[float]:
        if hasattr(self.m__model, 'coef_'):
            return float(self.m__model.coef_[-1])
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

        # Define shortcuts
        module_name = self.m__config_model['module']
        class_name = self.m__config_model['class']
        parameters = self.m__config_model['parameters']

        # Augment model parameters
        if class_name == 'RidgeCV':
            parameters['alphas'] = np.geomspace(1e-5, 1e5, 101)

        # Instantiate a new model according to the model_config
        model_class = get_class_by_name(module_name=module_name,
                                        class_name=class_name)
        self.m__model = model_class(**parameters)

        # Augment the sources: if we are using a forward model, we need to
        # add the planet signal as a new column to the sources here; if not,
        # we leave the sources unchanged
        if planet_signal is not None:
            sources = np.column_stack([sources, planet_signal.reshape(-1, 1)])

        # Fit model to the training data
        self.m__model.fit(X=sources, y=targets)

    def get_noise_prediction(self,
                             sources: np.ndarray,
                             add_dummy_column: bool = True) -> np.ndarray:
        """
        Get predictions of the "noise" part of the model.

        Args:
            sources: A 2D numpy array of shape (n_samples, n_features),
                which contains the data for which we want to make a
                prediction using the trained model.
            add_dummy_column: Whether or not to add a column of zeros
                to the sources. This is necessary if the model was
                trained using forward modeling (where the last column
                is the planet signal from the forward model, which we
                do not need to make a prediction about the noise).

        Returns:
            A 1D numpy array of shape (n_samples, ) containing the
            noise model predictions for the given inputs (sources).
        """

        # Only trained models can be used to make predictions
        if self.m__model is not None:

            # If requested, add a dummy column to the sources
            if add_dummy_column:
                dummy_column = np.zeros((sources.shape[0], 1))
                sources = np.column_stack([sources, dummy_column])
            
            # Return the noise model prediction
            return self.m__model.predict(X=sources)

        raise RuntimeError('You called get_noise_predictions() on an '
                           'untrained model!')
