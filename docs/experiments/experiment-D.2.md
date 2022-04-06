# Experiment D.2

Running this experiment basically consists of 2 steps:

1. For each combination of binning factor (in [1, 10, 100, 1000]) and observing conditions, you need to run the corresponding experiment in:
   ```text
   $HSR4HCI_EXPERIMENTS_DIR/appendix/D.2_residual-noise-with-and-without-oc/beta_pictoris__mp/experiments/<binning-factor>/<observing-conditions>
   ```
   There are 8 such combinations in total.
   Here is an example:
   ```bash
   python $HSR4HCI_SCRIPTS_DIR/experiments/single-script/01_run_pipeline.py \
     --experiment-dir $HSR4HCI_EXPERIMENTS_DIR/appendix/D.2_residual-noise-with-and-without-oc/beta_pictoris__mp/experiments/binning_factor-1000/with-oc
   ```
   You can either use the scripts in `single-script` or `multiple-scripts` for this.
3. Once you have run the experiment for all 8 combinations of binning factor and observation conditions, you can generate the corresponding result plots from the paper by running:
   ```bash
   python $HSR4HCI_EXPERIMENTS_DIR/appendix/D.2_residual-noise-with-and-without-oc/make_plots.py
   ```
   This will place the result plots in:
   ```text
   $HSR4HCI_EXPERIMENTS_DIR/appendix/D.2_residual-noise-with-and-without-oc/beta_pictoris__mp/plots
   ```
