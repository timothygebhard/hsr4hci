# General

This page contains information about the basic structure of the `experiments` directory, as well as additional information regarding how to run the experiments on a HTCondor-based computing cluster.


## Structure of the `experiments` directory

The `experiments` directory contains the configurations for all of our experiments. 
There are three main directories:

1. `appendix`: Experiments shown in Appendix A and D of our paper.
2. `main`: Experiments shown in the main part (Sect. 5 and 6) of our paper.
3. `demo`: A small demo experiment to help you familiarize yourself with our code base.

For some experiments (e.g., the mini-experiments from the appendix), the respective subdirectory will already contain all the scripts needed to run the experiment (e.g., a `make_plot.py` script).
This is also true for "evaluation scripts" which are only applicable to a specific experiment.

Experiments that use a workflow that can be recycled across various experiments (e.g., running our half-sibling regression algorithm) typically consist of a `config.json` file that specifies the exact experiment configuration (data set, algorithm, hyperparameters, ...). 
To run the experiments, we use the scripts in the `/hsr4hci/scripts/experiments` directory.

There are two main ways of running the half-sibling regression pipeline:
1. The version in `/hsr4hci/scripts/experiments/multiple-scripts`, where the pipeline is broken up into multiple scripts. 
   This is useful, for example, if you want to run experiments in parallel on a cluster. 
2. The version in `/hsr4hci/scripts/experiments/single-script`, which contains the entire pipeline in a single file.

To give a practical example, here is the command that you need to re-run the first experiment using HSR with signal fitting on the Beta Pictoris *L'* data set:

```bash
python <path-to-hsr4hci-repository>/scripts/experiments/single-script/01_run_pipeline.py --experiment-dir $HSR4HCI_EXPERIMENTS_DIR/main/5.1_first-results/signal_fitting/beta_pictoris__lp
```

More detailed information about how to run specific experiments can be found on the following pages.


## Running experiments with HTCondor

Scripts named `00_make_submit_files.py` can be used to create submission files for running our experiments on a [HTCondor-based cluster](https://htcondor.org/) with [DAGMan](https://research.cs.wisc.edu/htcondor/dagman/dagman.html).
