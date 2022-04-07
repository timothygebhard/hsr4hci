# General information

This page contains information about the basic structure of the `experiments` directory, as well as additional information regarding how to run the experiments on a HTCondor-based computing cluster.


## Structure of the `experiments` directory

The `experiments` directory contains the configurations for all of our experiments. 
There are three main directories:

1. `appendix`: Experiments shown in Appendix A and D of our paper.
2. `main`: Experiments shown in the main part (Sect. 5 and 6) of our paper.
3. `demo`: A small demo experiment to help you familiarize yourself with our code base.

For some experiments (e.g., the mini-experiments from the appendix), the respective subdirectory will already contain all the scripts needed to run the experiment (e.g., a `make_plot.py` script).
This is also true for "evaluation scripts" which are only applicable to a specific experiment.


## Experiment configurations

Every experiment essentially consists of a configuration file, `config.json`, which contains all the information about the data set, the parameters of the post-processing algorithm, and so on.
Here is an example of such a configuration file (taken from the [demo experiment](demo.md)):

```{eval-rst}
.. literalinclude:: ../../experiments/demo/config.json
   :language: json
```


## Running the HSR pipeline

To actually run an experiments, we use the scripts in the `$HSR4HCI_SCRIPTS_DIR/experiments` directory.
There are two main ways of running our half-sibling regression pipeline:

1. The version in `$HSR4HCI_SCRIPTS_DIR/experiments/single-script` contains the entire pipeline in a single file.
   Running an experiment is as easy as calling:
   ```bash
   python $HSR4HCI_SCRIPTS_DIR/experiments/single-script/01_run_pipeline.py \
     --experiment-dir /path/to/folder/with/experiment/config/file
   ```
2. The version in `$HSR4HCI_SCRIPTS_DIR/experiments/multiple-scripts` breaks up the pipeline into multiple scripts. 
   This is useful, for example, if you want to run experiments in parallel on a cluster (see below).
   In this case you need to run the following scripts in order:
   ```bash
   # (1) Train models
   python $HSR4HCI_SCRIPTS_DIR/experiments/multiple-scripts/01_train_models.py \
     --experiment-dir /path/to/folder/with/experiment/config/file ;
   
   # (2) Merge HDF files with residuals
   python $HSR4HCI_SCRIPTS_DIR/experiments/multiple-scripts/02_merge_hdf_files.py \
     --experiment-dir /path/to/folder/with/experiment/config/file ;
   
   # (3) Run stage 2
   python $HSR4HCI_SCRIPTS_DIR/experiments/multiple-scripts/03_run_stage_2.py \
     --experiment-dir /path/to/folder/with/experiment/config/file ;
   
   # (4) Merge FITS files with match fractions etc.
   python $HSR4HCI_SCRIPTS_DIR/experiments/multiple-scripts/04_merge_fits_files.py \
     --experiment-dir /path/to/folder/with/experiment/config/file ;
   
   # (5) Select residuals and construct signal estimate
   python $HSR4HCI_SCRIPTS_DIR/experiments/multiple-scripts/05_get_signal_estimate.py \
     --experiment-dir /path/to/folder/with/experiment/config/file ;
   ```
   Steps (1) and (3) can be parallelized by making use of the `--n-roi-splits` and `--roi-split` options, which cause the scripts to only process a certain part of the total region of interest (at these stage, all pixels can be processed independently).

In both cases, you might want to create the final result plot (as a PDF) by calling:
```bash
python $HSR4HCI_SCRIPTS_DIR/experiments/evaluate-and-plot/evaluate_and_plot_signal_estimate.py \
  --experiment-dir /path/to/folder/with/experiment/config/file
```

More detailed information about how to run the experiments in our paper can be found on the following pages.


(running-experiments-with-htcondor)=
## Running experiments with HTCondor

Some experiments are computationally rather expensive (hundreds to thousands of CPU hours). 
However, many parts of the HSR-method can be easily parallelized.
Therefore, when running the experiments to produce the results in the paper, we made use of a computational cluster based on [HTCondor](https://htcondor.org/) in combination with [DAGMan](https://research.cs.wisc.edu/htcondor/dagman/dagman.html) to handle the dependencies between the different steps of the pipeline.

Running experiments as a job on an HTCondor cluster requires a submission file which specifies the hardware requirements and the exact code / command to be run.
For simplicity and reproducibility, we have written scripts that can generate these submission files (and additionally any DAGMan files, if needed) automatically.
These scripts are typically named `00_make_submit_files.py`.

```{tip} 
If you are not working on an HTCondor-based cluster, you can safely ignore these scripts.
``` 

However, even if you are not using HTCondor, but want to run experiments on your own cluster infrastructure (e.g., Slurm), you might want to draw some inspiration from our approach 🙂 
