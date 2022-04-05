"""
Methods for importing things from modules based on their name.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Any

import importlib


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_member_by_name(module_name: str, member_name: str) -> Any:
    """
    Take the name of a module (e.g., ``sklearn.linear_model``) and the
    name of a member (e.g., a class or a function from that module;
    e.g., ``Ridge``) as the inputs, and try to import said member from
    the given module. If no such member exists in the module, an
    ``AttributeError`` is raised.

    Args:
        module_name: Name of the module from which to import.
        member_name: Name of the member to import.

    Returns:
        The specified module member, which can be virtually anything,
        that is, a variable, a function, a class, and so on.
    """

    # Load the module based on the given name. This will raise an
    # ImportError if the specified module cannot be loaded.
    _module = importlib.import_module(module_name)

    # Get the member based on the given name. This will raise an
    # AttributeError if the specified member cannot be found.
    _member = getattr(_module, member_name)

    return _member
