"""
Methods for creating HSR base models.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from copy import deepcopy
from functools import lru_cache
from typing import Any

import numpy as np

from hsr4hci.importing import get_member_by_name
from hsr4hci.typehinting import RegressorModel


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class BaseModelCreator:
    """
    Wrapper class for creating new base model instances.

    Example:
        >>> base_model_config = {
        >>>     'module': 'sklearn.linear_model',
        >>>     'class': 'LinearRegression',
        >>>     'parameters': {'fit_intercept': False},
        >>> }
        >>> bmc = BaseModelCreator(**base_model_config)
        >>> model = bmc.get_model_instance()
        >>> model
        LinearRegression(fit_intercept=False)

    .. note::

        Ideally, this function should simply take three arguments
        instead of a dictionary. The reason behind the current version
        is a poor early design choice for the experiment configuration
        files: The "class" parameter should have been called "name"
        instead, because ``class`` is a protected key word in Python
        that cannot be used as the name of an input parameter.
        However, changing this now would require updating all experiment
        configuration files and all training scripts...
    """

    def __init__(self, **base_model_config: Any) -> None:
        """
        Args:

            **base_model_config: A ``dict`` containing the configuration
                of the base model. It needs to have exactly three keys
                (see example above):
    
                - ``module``: A string with the module from which the
                  base model should be imported.
                - ``class``: A string with the class (= name) of the
                  base model.
                - ``parameters``: A dictionary with additional keyword
                  arguments that will be passed to the constructor of
                  ``module.class``. Can be empty: ``{}``.
        """

        # Unpack base model configuration
        self.module_name = base_model_config['module']
        self.class_name = base_model_config['class']
        self.parameters = base_model_config['parameters']

    @lru_cache(maxsize=1)
    def get_model_instance(self) -> RegressorModel:
        """
        Get a new instance of the base model defined in the config.

        Returns:
            An instance of a regression method (e.g., from ``sklearn``)
            that must provide the ``.fit()`` and ``.predict()`` methods.
        """

        # Get the model class and the model parameters
        model_class = get_member_by_name(
            module_name=self.module_name, member_name=self.class_name
        )
        model_parameters = deepcopy(self.parameters)

        # Augment the model parameters:
        # For RidgeCV models, we have to parse the ``alphas`` parameter (i.e.,
        # the regularization strengths) into a geometrically spaced array
        if (
            self.class_name in ('RidgeCV', 'LassoCV')
            and 'alphas' in model_parameters.keys()
        ):
            model_parameters['alphas'] = np.geomspace(
                *model_parameters['alphas']
            )

        # Instantiate a new model of the given class with the desired params
        model: RegressorModel = model_class(**model_parameters)

        return model
