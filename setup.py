"""
Setup script to install hsr4hci as a Python package.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from setuptools import setup


# -----------------------------------------------------------------------------
# RUN setup() FUNCTION
# -----------------------------------------------------------------------------

setup(
    name='hsr4hci',
    version='epsilon',
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
