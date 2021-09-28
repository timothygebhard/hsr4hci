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
        'astropy==4.2.1',
        'astroquery==0.4.1',
        'bottleneck==1.3.2',
        'click==7.1.2',
        'h5py==3.2.1',
        'joblib==1.0.1',
        'jupyter==1.0.0',
        'matplotlib==3.4.1',
        'numpy==1.20.2',
        'pandas==1.2.4',
        'photutils==1.1.0',
        'polarTransform==2.0.0',
        'python-dateutil==2.8.1',
        'requests==2.25.1',
        'scikit-image==0.18.1',
        'scikit-learn==0.24.2',
        'scipy==1.6.3',
        'seaborn==0.11.1',
        'tqdm==4.60.0',
    ],
    extras_require={
        'develop': [
            'coverage==5.5',
            'deepdiff==5.5.0',
            'flake8==3.9.1',
            'mypy==0.812',
            'pytest==6.2.3',
            'pytest-cov==2.11.1',
        ]
    },
    packages=find_packages(),
    zip_safe=False,
)
