# Create data and baseline

The scripts in this folder are used to create the data sets that we use for our experiments from a PynPoint data base file, as well as to compute the SNR baseline given by simple PCA-based PSF subtraction.

In particular, this folder contains the following files:

1. ``01a_create_datasets_from_hdf.py``:
   This script takes a PynPoint data base file as its input and creates several new HDF files from it containing several versions of the data with different levels of pre-stacking (i.e., block-wise combination of frames).
1. ``01b_create_datasets_from_fits.py``:
   This script takes a set of FITS files (containing the raw stack, parallactic angles, and the PSF template) its input and creates several new HDF files from it containing several versions of the data with different levels of pre-stacking (i.e., block-wise combination of frames).
3. ``02_run_pca.py``:
   This script will run the PCA-based PSF subtraction on the files created by ``01_create_datasets.py`` and create a ``signal_estimates.fits`` file for each of them.
   These FITS files contain the planet estimate based on the data that have been denoised with PCA.
4. ``03_compute_snrs.py``:
   This script will take the FITS files produced by ``02_run_pca.py`` and compute the signal-to-noise ratio (SNR) at the given (planet) positions as a function of the number of principal components.
   This allows us to find the optimal number of PCs for a given data set.
   The results are stored in a file ``figures_of_merit.csv``.
5. ``04_plot_snrs.py``:
   This script reads in the results in ``figures_of_merit.csv`` and plots the SNR as a function of the number of principal components, storing the resulting plot as ``snr_over_npc.pdf``.
6. ``example_config.json``:
   An example configuration file, showcasing all the expected information and options for the processing.
   A file with this structure (named ``config.json``) must be placed in the ``base_directory`` where the data processing is done (see below).
7. ``README.md``:
   This file.

All scripts take a ``--base-directory`` flag as an argument, which should be a path pointing to the directory in which the data processing takes place.
This directory must have the following structure:

```
base_directory/
|-- config.json [contains all information and options for this data set]
|-- raw/
|   |-- <pynpoint_database>.hdf ["raw" HDF file]
|-- processed/ [created by 01_create_datasets.py]
|   |-- stacked_1.hdf [pre-stacked data set with stacking factor 1]
|   |-- ...
|   |-- stacked_N.hdf [pre-stacked data set with stacking factor N]
|-- pca_baselines/ [created by 02_run_pca.py]
|   |-- stacked_1/
|   |   |-- signal_estimates.fits
|   |   |-- figures_of_merit.csv
|   |   |-- snr_over_npc.pdf
|   |-- ...
|   |-- stacked_N/
|   |   |-- ...
```
