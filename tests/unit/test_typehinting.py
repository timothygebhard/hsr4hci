"""
Tests for typehinting.py
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import pytest

from hsr4hci.typehinting import (
    RegressorModel,
    BaseLinearModel,
    BaseLinearModelCV,
)


# -----------------------------------------------------------------------------
# TEST CASES
# -----------------------------------------------------------------------------

def test__regressor_model() -> None:

    with pytest.raises(TypeError) as type_error:
        RegressorModel()  # type: ignore
    assert 'Protocols cannot be instantiated' in str(type_error)


def test__base_linear_model() -> None:

    with pytest.raises(TypeError) as type_error:
        BaseLinearModel()  # type: ignore
    assert 'Protocols cannot be instantiated' in str(type_error)


def test__base_linear_model_cv() -> None:

    with pytest.raises(TypeError) as type_error:
        BaseLinearModelCV()  # type: ignore
    assert 'Protocols cannot be instantiated' in str(type_error)
