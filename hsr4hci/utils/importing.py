"""
Utility functions for importing things from modules based on their name.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from typing import Callable

import importlib


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_member_by_name(module_name: str,
                       member_name: str) -> Callable:
    """
    Take the name of a module (e.g., 'sklearn.linear_model') and the
    name of a member (i.e., a class or a function from that module;
    e.g., 'Ridge') as the inputs, and try to import said member from
    the given module.
    If no such member exists in the module, an AttributeError is raised.

    Args:
        module_name: Name of the module from which to import.
        member_name: Name of the member to import.

    Returns:
        The specified member, which is a callable that can either be
         used directly (functions) or be instantiated (classes).
    """

    # Load the module based on the given name. This will raise an
    # ImportError if the specified module cannot be loaded.
    _module = importlib.import_module(module_name)

    # Get the member based on the given name. This will raise an
    # AttributeError if the specified member cannot be found.
    _member = getattr(_module, member_name)

    return _member
