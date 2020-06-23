# Create data and baseline

The scripts in this folder are used to create the data sets that we use for our experiments from a PynPoint data base file, as well as to compute the SNR baseline given by simple PCA-based PSF subtraction.

In particular, this folder contains the following files:

1. ``00_get_observing_conditions.py``:
   This script extracts the observing conditions from the original raw FITS files from the ESO archive.
2. ``01_plot_observing_conditions.py``:
   This script creates diagnostic plots for the observing conditions.
3. ``02_create_data_sets.py``:
   This script takes a PynPoint data base file or a set of pre-processed FITS files (containing the stack, the parallactic angles, and the unsatured PSF template) as its input and creates several new HDF files from it containing several versions of the data with different levels of pre-stacking (i.e., block-wise combination of frames).
4. ``03_run_pca.py``:
   This script will run the PCA-based PSF subtraction on the files created by ``02_create_datasets.py`` and create a ``signal_estimates.fits`` file for each of them.
   These FITS files contain the planet estimate based on the data that have been denoised with PCA.
5. ``04_compute_snrs.py``:
   This script will take the FITS files produced by ``03_run_pca.py`` and compute the signal-to-noise ratio (SNR) at the given (planet) positions as a function of the number of principal components.
   This allows us to find the optimal number of PCs for a given data set.
   The results are stored in a file ``figures_of_merit.csv``.
6. ``05_plot_pca_snr_over_npc.py``:
   This script reads in the results in ``figures_of_merit.csv`` and plots the SNR as a function of the number of principal components, storing the resulting plot as ``snr_over_npc.pdf``.
7. ``06_run_median_adi_and_compute_snr.py``:
   This script reads in runs a median ADI for each data set and computes the corresponding SNR for every planet.
8. ``README.md``:
   This file.

All scripts take a ``--base-directory`` flag as an argument, which should be a path pointing to the directory in which the data processing takes place.
This directory must have the following structure:

```
base_directory/
|-- config.json [contains all information and options for this data set]
|-- input/
|   |-- <pynpoint_database>.hdf ["raw" HDF file]
|   |-- <stack>.fits [instead of a PynPoint data base, the inputs can also be given as FITS files]
|   |-- <parang>.fits
|   |-- <psf_template>.fits [optional]
|-- processed/ [created by 02_create_datasets.py]
|   |-- psf_template.fits
|   |-- stacked_1.hdf [pre-stacked data set with stacking factor 1]
|   |-- stacked_1.fits
|   |-- ...
|   |-- stacked_N.hdf [pre-stacked data set with stacking factor N]
|   |-- stacked_N.fits
|-- pca_baselines/ [created by 03_run_pca.py]
|   |-- stacked_1/
|   |   |-- signal_estimates.fits
|   |   |-- figures_of_merit.csv
|   |   |-- snr_over_npc.pdf
|   |-- ...
|   |-- stacked_N/
|   |   |-- ...
```
