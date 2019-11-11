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
        self.m__has_planet_column = False

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
              add_planet_column: bool,
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
            add_planet_column: Whether or not to add the planet signal
                from the forward model as a column to the matrix of
                predictors. If this is False, the planet signal is only
                used to construct the sample weight or the train / test
                split when the `weight_mode` of the model is not set to
                "default".
            planet_signal: A 1D numpy array containing the planet signal
                time series (from forward modeling) to be included in the
                model. May be None if `use_forward_model` is False.
        """

        # ---------------------------------------------------------------------
        # Define some shortcuts to the model configuration
        # ---------------------------------------------------------------------

        module_name = self.m__config_model['module']
        class_name = self.m__config_model['class']
        parameters = self.m__config_model['parameters']
        weight_mode = self.m__config_model['weight_mode']

        # ---------------------------------------------------------------------
        # Augment model parameters
        # ---------------------------------------------------------------------

        # Increase the number of alpha values (i.e., regularization strengths)
        # for RidgeCV model. Adding them here  seems easier than adding 100
        # numbers to a configuration file.
        if class_name == 'RidgeCV':
            parameters['alphas'] = np.geomspace(1e-5, 1e5, 101)

        # ---------------------------------------------------------------------
        # Add planet_signal to the sources, if desired
        # ---------------------------------------------------------------------

        if add_planet_column and planet_signal is not None:

            # Add the planet_signal as an additional column to the sources
            sources = np.column_stack([sources, planet_signal.reshape(-1, 1)])

            # Keep track of the fact that this PixelPredictor was trained with
            # the planet signal from the forward model as a feature. This is
            # important when we later want to get the prediction of the "noise
            # part" of the model.
            self.m__has_planet_column = True

        # ---------------------------------------------------------------------
        # Instantiate a new model according to the model_config
        # ---------------------------------------------------------------------

        model_class = get_class_by_name(module_name=module_name,
                                        class_name=class_name)
        self.m__model = model_class(**parameters)

        # ---------------------------------------------------------------------
        # Fit model based on weight_mode
        # ---------------------------------------------------------------------

        # In the "default" weight_mode, the planet signal from the forward
        # model has no influence on the time steps that are used and we can
        # fit the model right away using the given sources and targets
        if weight_mode == 'default':
            self.m__model.fit(X=sources, y=targets)

        # For the "weighted" or "train_test" weight mode, we use the planet
        # signal from the forward model to decide which data points to use
        elif weight_mode in ('weighted', 'train_test'):

            # Standardize the planet signal such that is minimum value is 0
            # and its maximum value is 1.
            planet_signal_standardized = planet_signal - np.min(planet_signal)
            planet_signal_standardized /= np.max(planet_signal_standardized)

            # In the "weighted" mode, the weight of every time step t is given
            # by w(t) = 1 - standardized_planet_signal(t), meaning that frames
            # with lots of planet signal present (according to the current
            # forward model) contribute less to the fit of the model.
            # By paying less attention to the frames which contain planet
            # signal, we may be able to avoid the "noise part" of the model
            # picking up on it, which would then lead to self-subtraction.
            if weight_mode == 'weighted':

                # Compute the sample weights based on the planet signal
                sample_weight = 1 - planet_signal_standardized

                self.m__model.fit(X=sources,
                                  y=targets,
                                  sample_weight=sample_weight)

            # In the "train_test" mode, we only use those frames to fit the
            # model which -- according to the current forward model -- do not
            # contain any planet signal at all. This is basically identical to
            # the "weighted" case if we binarize the sample_weight (i.e., map
            # the weights to {0, 1}). Especially for small separations, this
            # weight_mode may reduce the number of used frames drastically.
            elif weight_mode == 'train_test':

                # Construct training sources and targets based on planet signal
                train_idx = np.where(planet_signal_standardized == 0)[0]
                train_sources = sources[train_idx, ...]
                train_targets = targets[train_idx]

                self.m__model.fit(X=train_sources, y=train_targets)

        else:
            raise ValueError(f'Illegal value for weight_mode: {weight_mode}')

    def get_noise_prediction(self,
                             sources: np.ndarray) -> np.ndarray:
        """
        Get predictions of the "noise part" of the model.

        Args:
            sources: A 2D numpy array of shape (n_samples, n_features),
                which contains the data for which we want to make a
                prediction using the trained model. Note: this array
                should NOT contain the planet signal from the forward
                model as its last column!

        Returns:
            A 1D numpy array of shape (n_samples, ) containing the
            noise model predictions for the given inputs (sources).
        """

        # Only trained models can be used to make predictions
        if self.m__model is not None:

            # If the model was trained using the planet signal from the
            # forward model as a feature, we need to add a "dummy column" to
            # the sources so that their shape matches the one expected by the
            # model. We choose this dummy column to be all zeros so that the
            # prediction is only made from the "noise part" of the model.
            if self.m__has_planet_column:
                dummy_column = np.zeros((sources.shape[0], 1))
                sources = np.column_stack([sources, dummy_column])

            # Return the noise model prediction
            return self.m__model.predict(X=sources)

        raise RuntimeError('You called get_noise_predictions() on an '
                           'untrained model!')
