# Experiment 6.4

Running this experiment is very similar to running [Experiment 5.1](experiment-5.1).
Basically, for every combination of algorithm (signal fitting, signal masking) and data set, you need to run the following two steps:

1. Run the respective experiment at
   ```text
   $HSR4HCI_EXPERIMENTS_DIR/main/6.4_observing-conditions/<algorithm>/<dataset>
   ```
   Here is an example:
   ```bash
   python $HSR4HCI_SCRIPTS_DIR/experiments/single-script/01_run_pipeline.py \
     --experiment-dir $HSR4HCI_EXPERIMENTS_DIR/main/6.4_observing-conditions/signal_fitting/beta_pictoris__lp
   ```
   You can either use the scripts in `single-script` or `multiple-scripts` for this.
2. Once you have run an experiment, you can create the corresponding result plot (containing the signal estimate as a PDF) by using the following command:
   ```bash
   python $HSR4HCI_SCRIPTS_DIR/experiments/evaluate-and-plot/evaluate_and_plot_signal_estimate.py \
     --experiment-dir $HSR4HCI_EXPERIMENTS_DIR/main/6.4_observing-conditions/<algorithm>/<dataset>
   ```
   The resulting plot will be stored in:
   ```text
   $HSR4HCI_EXPERIMENTS_DIR/main/6.4_observing-conditions/<algorithm>/<dataset>/plots
   ```

If you have run both experiment 5.1 and experiment 6.4 completely, you can run the following script to automatically create the LaTeX code for the comparison in Table 3 and print it to `stdout`:
```bash
python $HSR4HCI_EXPERIMENTS_DIR/main/6.4_observing-conditions/make_latex_table.py
```