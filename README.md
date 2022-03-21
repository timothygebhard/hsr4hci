# Half-Sibling Regression for High-Contrast Imaging

![Python 3.8 | 3.9](https://img.shields.io/badge/python-3.8_|_3.9-blue)
[![Checked with MyPy](https://img.shields.io/badge/mypy-checked-blue)](https://github.com/python/mypy)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![Tests](https://github.com/timothygebhard/hsr4hci/workflows/Tests/badge.svg?branch=master)
![Coverage Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/timothygebhard/40d8bf48dcbaf33c99e8de35ad6161f2/raw/hsr4hci.json)

---

This repository contains the code for all experiments and figures in our paper ***Half-sibling regression meets Direct Imaging: PSF modeling and subtraction using a flexible, domain knowledge-driven, causal framework***.

[TODO: Add link to paper once it is published.]

---

## ⚡ Quickstart

To get started, clone this repository and install `hsr4hci` as a Python package:

```
git clone git@github.com:timothygebhard/hsr4hci.git
cd hsr4hci
pip install .
```

If you want to use "developer options" (e.g., run unit tests), change the last line to:

```
pip install ".[develop]"
```

To run any experiments or reproduce our results, you will first need to download or create some data sets in the right format.
Please check out the [README file in the `datasets` directory](https://github.com/timothygebhard/hsr4hci/tree/master/datasets) for more detailed information on how to do this.


## 🧪 (Re)-running our experiments

All of our experiments can be found in the `experiments` directory.
For some experiments (e.g., the mini-experiments from the appendix), the respective subdirectory will already contain all the scripts needed to run the experiment (e.g., a `make_plot.py` script).
This is also true for "evaluation scripts" which are only applicable to a specific experiment.

Experiments that use a workflow that can be recycled across various experiments (e.g., running our half-sibling regression algorithm) typically consist of a `config.json` file that specifies the exact experiment configuration (data set, algorithm, hyperparameters, ...). 
To run the experiments, we use the scripts in the `scripts/experiments` directory.
There are two main ways of running the half-sibling regression pipeline:
1. The version in `scripts/experiments/multiple-scripts`, where the pipeline is broken up into multiple scripts. 
   This is useful, for example, if you want to run experiments in parallel on a cluster. 
2. The version in `scripts/experiments/single-script`, which contains the entire pipeline in a single file.

To give a practical example, here is the command that you need to re-run the first experiment using HSR with signal fitting on the Beta Pictoris L' data set:
```bash
python scripts/experiments/single-script/01_run_pipeline.py --experiment-dir experiments/01_first-results/signal_fitting/beta_pictoris__lp
```

Scripts named `00_make_submit_files.py` can be used to create submission files for running our experiments on a [HTCondor-based cluster](https://htcondor.org/) with [DAGMan](https://research.cs.wisc.edu/htcondor/dagman/dagman.html).

For more specific questions, please feel free to reach out directly to us!


## 🐭 Tests

This repository comes with an extensive set of unit and integration tests (based on [`pytest`](https://pytest.org)). 
After installing `hsr4hci` with the `[develop]` option, the tests can be run as:

```
pytest tests
```

You can also use these tests to ensure that the code is compatible with newer versions of the libraries than the one in `setup.py`.


## 📜 Cite this work

To cite this work, feel free to use the following BibTeX entry:

```
[TODO: Add BibTeX entry]
```


## ⚖️ License and copyright

The code in this repository is property of the [Max Planck Society](https://www.mpg.de/en).

[TODO: Add license information.]