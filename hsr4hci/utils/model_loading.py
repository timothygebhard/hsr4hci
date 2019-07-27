"""
Provide functions for loading models.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import importlib


# -----------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# -----------------------------------------------------------------------------

def get_class_by_name(module_name: str,
                      class_name: str):
    """
    Take a module name (e.g., 'sklearn.linear_model') and a class name
    (e.g., 'Ridge') as the inputs, and tries to import said class from
    the given module.
    If no such class exists in the module, an AttributeError is raised.

    Args:
        module_name: Name of the module from which to import the class.
        class_name: Name of the class to import.

    Returns:
        The specified class (which can then be instantiated).
    """
    
    # Load the module based on the given name. This will raise an ImportError
    # if the specified module cannot be loaded.
    _module = importlib.import_module(module_name)
    
    # Get the class based on the given name. This will raise an AttributeError
    # if the specified class cannot be found.
    _class = getattr(_module, class_name)
    
    return _class
