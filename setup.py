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

setup(name='hsr4hci',
      version='epsilon',
      description='hsr4hci: Half-Sibling Regression for High-Contrast Imaging',
      url='https://github.com/timothygebhard/hsr4hci',
      install_requires=[
          'astropy',
          'bottleneck',
          'contexttimer',
          'h5py',
          'joblib',
          'jupyter',
          'matplotlib',
          'numpy',
          'pandas',
          'photutils',
          'pytest',
          'scikit-image',
          'scikit-learn',
          'scipy',
          'tqdm',
      ],
      packages=['hsr4hci'],
      zip_safe=False,
      entry_points={
        'console_scripts': [
            'compute_snr = scripts.compute_snr:main',
        ]},
      )
