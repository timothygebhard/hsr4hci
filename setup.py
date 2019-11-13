from setuptools import setup

setup(name='hsr4hci',
      version='epsilon',
      description='hsr4hci: Half-Sibling Regression for High-Contrast Imaging',
      url='https://github.com/timothygebhard/hsr4hci',
      install_requires=['astropy',
                        'cvxpy',
                        'h5py',
                        'joblib',
                        'jupyter',
                        'matplotlib',
                        'numpy',
                        'pandas',
                        'photutils',
                        'pynpoint',
                        'scikit-image',
                        'scikit-learn',
                        'scipy',
                        'tqdm'],
      packages=['hsr4hci'],
      zip_safe=False)
