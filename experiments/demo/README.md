# 🧪 Demo experiment

This folder contains a small demo experiment to help you get started with using the methods from this repository.
It is essentially a simplified version of experiment 5.1 from our paper that uses a smaller temporal grid and a heavily binned version of our Beta Pictoris *L'* data set to make sure it runs quickly even on a single machine.

1. Before you get started, make sure that `hsr4hci` and its dependencies are installed (ideally in a [virtual environment](https://virtualenv.pypa.io/en/latest/) to avoid any version conflicts between Python packages). 
2. If you have not done so already, [download the data sets](https://doi.org/10.17617/3.LACYPN) and place them in the right (sub)-directories, as described [here](https://github.com/timothygebhard/hsr4hci/tree/master/datasets).
3. Now, to start and run the experiment, use:
   ```
   python <path>/hsr4hci/scripts/experiments/single-script/01_run_pipeline.py --experiment-dir <path>/experiments/demo 
   ```
   (Replace the `<path>` with the correct path to the `hsr4hci` repository.) 
4. Depending on your hardware, the experiment should take about 10 to 20 minutes to run.
   (If you find this too long, you can, for example, open the `config.json` and try to increase the `binning_factor` further.)
5. Once they experiment has finished, you can find the results in `./plots`.
   The signal estimate is stored as a FITS file, which you can view, for example, using [SAOImageDS9](https://sites.google.com/cfa.harvard.edu/saoimageds9).
6. You can also generate a plot as a PDF by calling the following script:
   ```
   python <path>/scripts/experiments/evaluate-and-plot/evaluate_and_plot_signal_estimate.py --experiment-dir ./experiments/demo
   ```
   You will find the final result plot in `./plots`.

---

Feel free to use this as a starting point to explore our code in more detail. 
We have tried our best to document and comment everything, and we hope it should not be too hard to find your way around.