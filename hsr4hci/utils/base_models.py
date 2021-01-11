"""
Utility functions for dealing with HSR base models.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from copy import deepcopy
from functools import lru_cache
from typing import Any

import numpy as np

from hsr4hci.utils.importing import get_member_by_name
from hsr4hci.utils.typehinting import RegressorModel


# -----------------------------------------------------------------------------
# CLASS DEFINITIONS
# -----------------------------------------------------------------------------

class BaseModelCreator:

    def __init__(self, **base_model_config: Any) -> None:

        # Unpack base model configuration
        self.module_name = base_model_config['module']
        self.class_name = base_model_config['class']
        self.parameters = base_model_config['parameters']

    @lru_cache(maxsize=1)
    def get_model_instance(self) -> RegressorModel:
        """
        Get a new instance of the base model defined in the config.

        Returns:
            An instance of a regression method (e.g., from sklearn) that
            must provide the .fit() and .predict() methods.
        """

        # Get the model class and the model parameters
        model_class = get_member_by_name(
            module_name=self.module_name, member_name=self.class_name
        )
        model_parameters = deepcopy(self.parameters)

        # Augment the model parameters:
        # For RidgeCV models, we have to parse the `alphas` parameter (i.e.,
        # the regularization strengths) into a geometrically spaced array
        if self.class_name == 'RidgeCV':
            model_parameters['alphas'] = np.geomspace(
                *model_parameters['alphas']
            )

        # Instantiate a new model of the given class with the desired params
        model: RegressorModel = model_class(**model_parameters)

        return model
