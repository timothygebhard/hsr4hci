"""
Tests for base_models.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from sklearn.linear_model import RidgeCV

from hsr4hci.base_models import BaseModelCreator


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__base_model_creator() -> None:

    base_model_creator = BaseModelCreator(
        **{
            'module': 'sklearn.linear_model',
            'class': 'RidgeCV',
            'parameters': {
                'fit_intercept': True,
                'alphas': [1, 100, 3],
            },
        }
    )
    model = base_model_creator.get_model_instance()

    assert isinstance(model, RidgeCV)
    assert model.fit_intercept
    assert set(model.alphas) == {1, 10, 100}
