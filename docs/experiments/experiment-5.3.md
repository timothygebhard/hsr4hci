# Experiment 5.3

```{caution}
Running this experiment is somewhat complicated, because it actually consists of a large number of sub-experiments:
Basically, every field in the table-like result plot consists of 6 individual experiments.
```

This experiment requires several steps:

1. For each data set, you need to first create the respective sub-experiments for each combination of contrast, separation and azimuthal position of the injected companion.
   To do this, you need to run the following script:
   ```bash
   python $HSR4HCI_EXPERIMENTS_DIR/main/5.3_photometry-artificial-planets/00_create_experiments.py \
     --directory $HSR4HCI_EXPERIMENTS_DIR/main/5.3_photometry-artificial-planets/<dataset>/<algorithm>
   ```
   This will create the experiment folders (by default: 630 + 1) in:
   ```text
   $HSR4HCI_EXPERIMENTS_DIR/main/5.3_photometry-artificial-planets/<dataset>/<algorithm>/experiments
   ```
   Additionally, it will also create the submission files that you need to run the experiment with HTCondor and DAGMan.
   (If you are not using HTCondor, you can just ignore these files.)
2. Run each of the 631 experiments.
   For this, you need to use the scripts in `$HSR4HCI_SCRIPTS_DIR/experiments/brightness-ratio`, as they will take care of injecting an artificial companion at the right position and with the correct contrast.

   For example, you will need to run:
   ```bash
   # HSR
   python $HSR4HCI_SCRIPTS_DIR/experiments/brightness-ratio/run_hsr.py \
     --experiment-dir $HSR4HCI_EXPERIMENTS_DIR/main/5.3_photometry-artificial-planets/beta_pictoris__lp/signal_fitting/experiments/10.00__2.0__a
   
   # PCA
   python $HSR4HCI_SCRIPTS_DIR/experiments/brightness-ratio/run_pca.py \
     --experiment-dir $HSR4HCI_EXPERIMENTS_DIR/main/5.3_photometry-artificial-planets/beta_pictoris__lp/pca-20/experiments/10.00__2.0__a
   ```
3. Once you have run all experiments for a given combination of `algorithm` and `dataset`, you need collect the results (i.e., run the actual throughput computation):
   ```bash
   python $HSR4HCI_EXPERIMENTS_DIR/main/5.3_photometry-artificial-planets/01_collect_results.py \
     --directory $HSR4HCI_EXPERIMENTS_DIR/main/5.3_photometry-artificial-planets/<dataset>/<algorithm> \
     --n-jobs 32
   ```
   The `--n-jobs` argument controls the number of parallel processes used to collect the results.
   This is independent of any potential parallelization on a cluster.

   Running this script will create a results TSV file:
   ```text
   $HSR4HCI_EXPERIMENTS_DIR/main/5.3_photometry-artificial-planets/<dataset>/<algorithm>/results__classic.tsv
   ```
4. Once you have created the TSV file that contains the logFPF and throughput for every combination of contrast, separation and azimuthal position, you can create the table-like result plot from Figure 9 by running:
   ```bash
   python $HSR4HCI_EXPERIMENTS_DIR/main/5.3_photometry-artificial-planets/02_make_plot.py \
     --directory $HSR4HCI_EXPERIMENTS_DIR/main/5.3_photometry-artificial-planets/<dataset>/<algorithm>
   ```
5. Finally, if you have run all experiments for a given data set (i.e., for all algorithms), you can create the additional plot that illustrates the computation of the contrast curve (Figure 8) as follows:
   ```bash
   python $HSR4HCI_EXPERIMENTS_DIR/main/5.3_photometry-artificial-planets/03_plot_interpolated_fpf.py \
     --dataset <dataset>
   ```
   The resulting plot will be stored in:
   ```text
   $HSR4HCI_EXPERIMENTS_DIR/main/5.3_photometry-artificial-planets/<dataset>
   ```
6. To create the plot from Figure 13 (i.e., the plot that compares all contrast curves), run:
   ```bash
   python $HSR4HCI_EXPERIMENTS_DIR/main/5.3_photometry-artificial-planets/04_plot_all_contrast_curves.py \
     --dataset <dataset>
   ```
   The resulting plot will also be stored in:
   ```text
   $HSR4HCI_EXPERIMENTS_DIR/main/5.3_photometry-artificial-planets/<dataset>
   ```
