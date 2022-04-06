# Experiment 5.4

To run this experiment, the following steps are required:

1. For every combination of algorithm (signal fitting and signal masking), data set, and binning factor, run the experiment using the scripts in `HSR4HCI_SCRIPTS_DIR/experiments/hypothesis-based` (to use the hypothesis-based version of the HSR):
   ```bash
   python HSR4HCI_SCRIPTS_DIR/experiments/hypothesis-based/01_train_based_on_hypothesis.py \
     --experiment-dir $HSR4HCI_EXPERIMENTS_DIR/main/5.4_photometry-real-planets/<algorithm>/<dataset>/<binning-factor>
   ```
   If you have access to a cluster, you can parallelize this using the `--n-roi-splits` and `--roi-split` options.
   This step will create a directory:
   ```text
   $HSR4HCI_EXPERIMENTS_DIR/main/5.4_photometry-real-planets/<algorithm>/<dataset>/<binning-factor>/fits/partial
   ```
   which contains the partial result files.
   If you do not parallelize the experiment using `--n-roi-splits` and `--roi-split`, this folder will only contain a single FITS file.
2. Merge all partial result files using the following command:
   ```bash
   python $HSR4HCI_SCRIPTS_DIR/experiments/hypothesis-based/02_merge_residuals_and_get_signal_estimate.py \
     --experiment-dir $HSR4HCI_EXPERIMENTS_DIR/main/5.4_photometry-real-planets/<algorithm>/<dataset>/<binning-factor>
   ```
   This will create a `residuals.fits` and a `hypotheses.fits` file at the same level as the `partial` folder.
   ```{attention}
   This step is necessary even if you did not parallelize the experiment using `--n-roi-splits` and `--roi-split`.
   ```
3. Once you have run all 18 combinations of algorithm, data set and binning factor, you can generate the final result table from Table 2 as follows:
   ```bash
   python $HSR4HCI_EXPERIMENTS_DIR/main/5.4_photometry-real-planets/get_contrast_estimates.py
   ```
   This will create a TSV file with the estimated contrasts and a LaTeX file with the table from the paper.
   Both files will be placed in:
   ```text
   $HSR4HCI_EXPERIMENTS_DIR/main/5.4_photometry-real-planets
   ```
