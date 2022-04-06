# Experiment A.2

To run experiment A.2, you first need to create some FITS files that contain the correlation coefficient maps.
To do this, run the following command:

```bash
python $HSR4HCI_EXPERIMENTS_DIR/appendix/A.2_correlation-coefficient-maps/run_experiment.py
```

The resulting FITS files are stored in:
```text
$HSR4HCI_EXPERIMENTS_DIR/appendix/A.2_correlation-coefficient-maps/fits
```

To create the plots shown in the paper, you need to run another script:

```bash
python $HSR4HCI_EXPERIMENTS_DIR/appendix/A.2_correlation-coefficient-maps/make_plots.py
```

The corresponding plots (PDF files) are then stored in:
```text
$HSR4HCI_EXPERIMENTS_DIR/appendix/A.2_correlation-coefficient-maps/plots
```
