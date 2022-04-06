# Scripts

The `scripts` directory consists of two subdirectories:

1. `experiments`: Scripts in this directory contain code that can be used to run experiments, that is, basically apply the HSR method to a given data set and evaluate the results.
2. `figures`: Scripts in this directory were only used to create illustrative figures for the paper.

Here are some additional details about the scripts in the `experiments` folder:

* `brightness-ratio`: Scripts for running the experiments in Sect. 5.3 (inject artificial companions and run a PCA- or HSR-based post-processing pipeline).
* `evaluate-and-plot`: Scripts for computing the logFPF and other metrics after running HSR / PCA, and for plotting the signal estimates (or intermediate data products).
  Usually requires running scripts from `multiple-scripts`, `run-pca`, or `single-script` first.
* `hypothesis-based`: Scripts for running the HSR in hypothesis-based mode (see Sect. 3.4 in the paper). 
  We used these scripts only for the experiments in Sect. 5.4.
* `multiple-scripts`: Scripts for running the different steps of an HSR pipeline in separate steps. 
   We used these scripts to run experiments on a cluster where we wanted to parallelize some stages (e.g., training) over many nodes.
* `pixel-coefficients`: Scripts for running the experiments in Sect. 5.2.
* `run-pca`: Scripts for running a PCA-based post-processing pipeline (e.g., for the baseline results in Fig. 6)
* `single-script`: Scripts for running an HSR-based post-processing pipeline in a single step.
