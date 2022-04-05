"""
Configuration file for the Sphinx documentation builder.
"""

# -----------------------------------------------------------------------------
# IMPORT
# -----------------------------------------------------------------------------

from os.path import join, dirname


# -----------------------------------------------------------------------------
# PROJECT INFORMATION
# -----------------------------------------------------------------------------

# Project name, copyright and author
# noinspection PyShadowingBuiltins
copyright = (
    u'2022 by Max Planck Society / '
    u'Max Planck Institute for Intelligent Systems'
)
author = u'Timothy Gebhard'
project = u'hsrh4ci'

# The short X.Y version
with open(join(dirname(__file__), "../hsr4hci/VERSION")) as version_file:
    version = version_file.read().strip()

# The full version, including alpha/beta/rc tags
release = version


# -----------------------------------------------------------------------------
# GENERAL CONFIGURATION
# -----------------------------------------------------------------------------

# Sphins extensions
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinx_math_dollar',
]

# The master toctree document.
master_doc = 'index'

# Do not expand default arguments
# See: https://stackoverflow.com/a/67482867/4100721
autodoc_preserve_defaults = True

# -----------------------------------------------------------------------------
# OPTIONS FOR HTML OUTPUT
# -----------------------------------------------------------------------------

# HTML theme
html_theme = "furo"

# Additional theme options
html_theme_options = {}

# Options for MathJax (in particular: sphinx-math-dollar)
mathjax3_config = {
    'tex2jax': {
        'inlineMath': [["\\(", "\\)"]],
        'displayMath': [["\\[", "\\]"]],
    },
}

# -----------------------------------------------------------------------------
# OPTIONS FOR INTERSPHINX
# -----------------------------------------------------------------------------

# Mappings to documentations of other packages
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'astropy': ('https://docs.astropy.org/en/stable', None),
    'h5py': ('https://docs.h5py.org/en/latest/', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'pynpoint': ('https://pynpoint.readthedocs.io/en/latest/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# -----------------------------------------------------------------------------
# OPTIONS FOR AUTODOC
# -----------------------------------------------------------------------------

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"
