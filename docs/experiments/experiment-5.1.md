# Experiment 5.1

Running this experiment is relatively straight-forward.
Basically, for every combination of algorithm (signal fitting, signal masking, and PCA) and data set, you need to run the following two steps:

1. Run the respective experiment at
   ```text
   $HSR4HCI_EXPERIMENTS_DIR/main/5.1_first-results/<algorithm>/<dataset>
   ```
   Here is an example:
   ```bash
   python $HSR4HCI_SCRIPTS_DIR/experiments/single-script/01_run_pipeline.py \
     --experiment-dir $HSR4HCI_EXPERIMENTS_DIR/main/5.1_first-results/signal_fitting/beta_pictoris__lp
   ```
   For HSR, you can either use the scripts in `single-script` or `multiple-scripts` for this.
   For PCA, use the script in `$HSR4HCI_SCRIPTS_DIR/experiments/run-pca`.
2. Once you have run an experiment, you can create the corresponding result plot (containing the signal estimate as a PDF) by using the following command:
   ```bash
   python $HSR4HCI_SCRIPTS_DIR/experiments/evaluate-and-plot/evaluate_and_plot_signal_estimate.py \
     --experiment-dir $HSR4HCI_EXPERIMENTS_DIR/main/5.1_first-results/<algorithm>/<dataset>
   ```
   The resulting plot will be stored in:
   ```text
   $HSR4HCI_EXPERIMENTS_DIR/main/5.1_first-results/<algorithm>/<dataset>/plots
   ```
3. Optionally, to create the kind of plots from Figure 4 in our paper, you can also run:
   ```bash
   python $HSR4HCI_SCRIPTS_DIR/experiments/evaluate-and-plot/plot_results_of_stage_2.py \
     --experiment-dir $HSR4HCI_EXPERIMENTS_DIR/main/5.1_first-results/<algorithm>/<dataset>
   ```
   The plots will be placed in the same `plots` directory as the plot of the signal estimate.