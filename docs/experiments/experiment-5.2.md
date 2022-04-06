# Experiment 5.2

Running this experiment requires the following steps:

1. For each data set, run the respective experiment using the following script / command:
   ```bash
   python $HSR4HCI_SCRIPTS_DIR/experiments/pixel-coefficients/01_train_models.py \
     --experiment-dir $HSR4HCI_EXPERIMENTS_DIR/main/5.2_pixel-coefficients/<dataset>/binning-factor_1
   ```
   If you have access to a cluster, you can parallelize this using the `--n-roi-splits` and `--roi-split` options.
   This step will create a directory:
   ```text
   $HSR4HCI_EXPERIMENTS_DIR/main/5.2_pixel-coefficients/<dataset>/binning-factor_1/fits/partial
   ```
   which contains the partial result files.
   If you do not parallelize the experiment using `--n-roi-splits` and `--roi-split`, this folder will only contain a single FITS file.
2. Merge all partial result files using the following command:
   ```bash
   python $HSR4HCI_SCRIPTS_DIR/experiments/pixel-coefficients/02_merge_fits_files.py \
     --experiment-dir $HSR4HCI_EXPERIMENTS_DIR/main/5.2_pixel-coefficients/<dataset>/binning-factor_1
   ```
   This will create a `coefficients.fits` file at the same level as the `partial` folder.
   ```{attention}
   This step is necessary even if you did not parallelize the experiment.
   In this case, the "merging" step basically just copies and renames the result FITS file to the expected location.
   ```
3. Finally, you can create the plots from the paper by running:
   ```bash
   python $HSR4HCI_EXPERIMENTS_DIR/main/5.2_pixel-coefficients/make_plots.py \
     --experiment-dir $HSR4HCI_EXPERIMENTS_DIR/main/5.2_pixel-coefficients/<dataset>/binning-factor_1
   ```
   The plots are stored in a subdirectory of:
   ```text
   $HSR4HCI_EXPERIMENTS_DIR/main/5.2_pixel-coefficients/plots
   ```