# Experiment D.1

```{caution}
Running this experiment is somewhat complicated, because it actually consists of a large number of sub-experiments:
Basically, every point in the final figure is one experiment.
```

This experiment requires several steps:

1. First, you need to _create_ the experiment directories; one for each combination of `algorithm` (signal fitting and signal masking with and without observing conditions, PCA with $n$ components) and `binning_factor`.
   Additionally, you can repeat the whole experiment for different data sets.
   To create the experiment folders for a given combination, you need to run the following command:
   ```bash
   python $HSR4HCI_EXPERIMENTS_DIR/appendix/D.1_fpf-as-function-of-temporal-binning/00_create_experiments.py \
     --dataset beta_picturis__lp \
     --algorithm signal_fitting__oc
   ```
   This is just an example; as mentioned, you will have to repeat this for different values of `dataset` and `algorithm`.
   For each data set, running this command will create a new subdirectory in the `$HSR4HCI_EXPERIMENTS_DIR/appendix/D.1_fpf-as-function-of-temporal-binning` with the name of the respective data set.
   These subdirectories will again contain subdirectories for each algorithm.

   If you run the above command for all combinations in the paper, you should get the following structure:
   ```text
   |--beta_pictoris__lp
   |  |--signal_fitting__oc
   |  |--signal_fitting
   |  |--signal_masking
   |  |--signal_masking__oc
   |  |--pca
   |--beta_pictoris__mp
   |  |--signal_fitting__oc
   |  |--signal_masking
   |  |--pca
   |  |--signal_fitting
   |  |--signal_masking__oc
   |--r_cra__lp
   |  |---signal_fitting__oc
   |  |---signal_masking
   |  |---pca
   |  |---signal_fitting
   |  |---signal_masking__oc
   ```
2. Each of the directories above (i.e., each combination of data set and algorithm) will *again* contain a set of subdirectories, one for each binning factor. 
   This should look as follows:
   ```text
    00_create_submit_files.sh
    01_start_jobs.sh
    binning_factor-0001 -> $HSR4HCI_EXPERIMENTS_DIR/main/5.1_first-results/<algorithm>/<dataset>
    binning_factor-0002
    binning_factor-0003
    binning_factor-0005
    binning_factor-0007
    binning_factor-0011
    binning_factor-0015
    binning_factor-0022
    binning_factor-0031
    binning_factor-0044
    binning_factor-0063
    binning_factor-0089
    binning_factor-0126
    binning_factor-0178
    binning_factor-0251
    binning_factor-0355
    binning_factor-0502
    binning_factor-0709
    binning_factor-1002
    binning_factor-1415
    binning_factor-2000
   ```
   The two shell script files (`00_create_submit_files.sh` and `01_start_jobs.sh`) are only relevant if you are running the experiment on an HTCondor cluster; otherwise, you can ignore them.
   
3. The `binning_factor-0001` experiment is just a symlink to the respective part of experiment 5.1 (to prevent running the same experiment multiple times).
   If you have not yet run that experiment, now would be a good time.
   All other directories are "normal" experiment directories, that is, they contain a `config.json` that describes the experiment.
4. For each `binning_factor` (and, of course, each `algorithm` and `dataset` ...), you need to run the HSR (or PCA) pipeline.
   For example, you want to run something like this:
   ```bash
   # HSR
   python $HSR4HCI_SCRIPTS_DIR/experiments/single-script/01_run_pipeline.py --experiment-dir $HSR4HCI_EXPERIMENTS_DIR/appendix/D.1_fpf-as-function-of-temporal-binning/<dataset>/<algorithm>/<binning-factor>
   
   # PCA
   python $HSR4HCI_SCRIPTS_DIR/experiments/run-pca/01_run_pca.py --experiment-dir $HSR4HCI_EXPERIMENTS_DIR/appendix/D.1_fpf-as-function-of-temporal-binning/<dataset>/<algorithm>/<binning-factor>  
   ```
   In practice, it is probably useful to do all this programmatically (i.e., via a script).
5. Once you have run all the experiments (i.e., all combinations of algorithm, binning factor, and data set), you need to "collect the results", that is, run the evaluation procedure to compute the logFPF score for every experiment.
   For each experiment, you need to run the following command:
   ```bash
   python $HSR4HCI_EXPERIMENTS_DIR/appendix/D.1_fpf-as-function-of-temporal-binning/01_collect_results.py \
     --dataset <dataset> \
     --algorithm signal_fitting__oc \ 
     --n-jobs 8
   ```
   The parameter `--n-jobs` controls the number of parallel processes that are used to collect the results (this is independent of a potential parallelization on a cluster).
   
   Running this command will create a file `metrics__<planet>.tsv` (where `<planet>` is `b` for all experiments in the paper) in `$HSR4HCI_EXPERIMENTS_DIR/appendix/D.1_fpf-as-function-of-temporal-binning/<dataset>/<algorithm>`.
   These TSV files contain the logFPF score for every binning factor.
7. Once you have collected the results of all experiments, you can create the final plot by running:
   ```bash
   python $HSR4HCI_EXPERIMENTS_DIR/appendix/D.1_fpf-as-function-of-temporal-binning/02_plot_logfpf_over_binning_factor.py \
     --dataset <dataset>
   ```
   The plot will be placed in:
   ```text
   $HSR4HCI_EXPERIMENTS_DIR/appendix/D.1_fpf-as-function-of-temporal-binning/<dataset>
   ```
