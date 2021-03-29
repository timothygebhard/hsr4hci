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
        'astropy',
        'astroquery',
        'bottleneck',
        'contexttimer',
        'h5py',
        'joblib',
        'jupyter',
        'matplotlib>=3.3.2',
        'numpy',
        'pandas',
        'peakutils',
        'photutils',
        'pynpoint==0.9.0',
        'pytest',
        'python-dateutil',
        'requests',
        'scikit-image',
        'scikit-learn',
        'scipy',
        'seaborn',
        'tqdm',
    ],
    extras_require={
        'develop': [
            'coverage>=5.3',
            'deepdiff>=5.0.2',
            'flake8>=3.8.3',
            'mypy>=0.782',
            'pytest>=6.0.2',
            'pytest-cov>=2.10.1',
        ]
    },
    packages=find_packages(),
    zip_safe=False,
)
