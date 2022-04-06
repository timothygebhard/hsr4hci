"""
Setup script to install hsr4hci as a Python package.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from os.path import join, dirname
from setuptools import find_packages, setup


# -----------------------------------------------------------------------------
# RUN setup() FUNCTION
# -----------------------------------------------------------------------------

# Get version from VERSION file
with open(join(dirname(__file__), "hsr4hci/VERSION")) as version_file:
    version = version_file.read().strip()

# Run setup()
setup(
    name='hsr4hci',
    version=version,
    description='hsr4hci: Half-Sibling Regression for High-Contrast Imaging',
    url='https://github.com/timothygebhard/hsr4hci',
    install_requires=[
        'astropy~=5.0.1',
        'astroquery~=0.4.5',
        'bottleneck~=1.3.2',
        'click~=8.0.3',
        'h5py~=3.6.0',
        'joblib~=1.1.0',
        'jupyter~=1.0.0',
        'matplotlib~=3.5.1',
        'numpy~=1.21.4',
        'pandas~=1.3.5',
        'photutils~=1.2.0',
        'polarTransform~=2.0.0',
        'python-dateutil~=2.8.2',
        'requests~=2.26.0',
        'scikit-image~=0.19.1',
        'scikit-learn~=1.0.1',
        'scipy~=1.7.3',
        'seaborn~=0.11.2',
        'tqdm~=4.62.3',
        'typing-extensions~=4.0.1',
        'wheel~=0.37.1',
    ],
    extras_require={
        'develop': [
            'coverage~=6.2',
            'deepdiff~=5.6.0',
            'flake8~=4.0.1',
            'mypy~=0.920',
            'pytest~=6.2.5',
            'pytest-cov~=3.0.0',
        ],
        'docs': [
            'furo',
            'myst-parser',
            'pygments-pytest',
            'sphinx',
            'sphinx-copybutton',
            'sphinx-math-dollar',
        ]
    },
    packages=find_packages(),
    zip_safe=False,
)
