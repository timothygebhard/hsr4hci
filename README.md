# Half-Sibling Regression for High-Contrast Imaging

![Python 3.8 | 3.9](https://img.shields.io/badge/python-3.8_|_3.9-blue)
[![Checked with MyPy](https://img.shields.io/badge/mypy-checked-blue)](https://github.com/python/mypy)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![Tests](https://github.com/timothygebhard/hsr4hci/workflows/Tests/badge.svg?branch=master)
![Coverage Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/timothygebhard/40d8bf48dcbaf33c99e8de35ad6161f2/raw/hsr4hci.json)

---

This repository contains the code for all experiments and figures in our paper ***Half-sibling regression meets exoplanet imaging: PSF modeling and subtraction using a flexible, domain knowledge-driven, causal framework***.

**[TODO: Add link to paper once it is published.]**

---

## ⚡ Getting started

The code in this repository is organized as a Python package named `hsr4hci` together with a set of scripts that use the functions and classes of the package.
To get started, clone this repository and install `hsr4hci` as a Python package:

```
git clone git@github.com:timothygebhard/hsr4hci.git
cd hsr4hci
pip install -e .
```

The `-e` option installs the package in "edit mode", which ensures that runs directly from the folder that you got by cloning this repository, instead of being copied to the `site-package` of your Python installation (a location where you usually would not want to store, e.g., data set files).

If you want to use "developer options" (e.g., run unit tests), change the last line to:

```
pip install -e ".[develop]"
```

**Note:** The code was written for Python 3.8 and above; earlier versions will likely require some small modifications.


## 🪐 Data sets

To run any experiments or reproduce our results, you will first need to [download](https://doi.org/10.17617/3.LACYPN) or create some data sets in the right format.
Please check out the [README file in the *datasets* directory](https://github.com/timothygebhard/hsr4hci/tree/master/datasets) for more detailed information on how to do this.


## 🧪 (Re)-running our experiments

All of our experiments can be found in the `experiments` directory, where we keep [another README](https://github.com/timothygebhard/hsr4hci/tree/master/experiments) with more instructions for how to replicate our results.
A good starting point if you are just getting started could the to run the demo experiment that we have prepared in the [*demo* directory](https://github.com/timothygebhard/hsr4hci/tree/master/experiments/demo).

## 🐭 Tests

This repository comes with an extensive set of unit and integration tests (based on [`pytest`](https://pytest.org)). 
After installing `hsr4hci` with the `[develop]` option, the tests can be run as:

```
pytest tests
```

You can also use these tests to ensure that the code is compatible with newer versions of the libraries than the one in `setup.py`.


## 📜 Citing this work

To cite this work, feel free to use the following BibTeX entry:

```
[TODO: Add BibTeX entry]
```


## ✏️ Authors

All code was written by Timothy Gebhard, with some contributions from Markus Bonse.


## ⚖️ License and copyright

The code in this repository is property of the [Max Planck Society](https://www.mpg.de/en).

We are releasing it under a BSD-3 Clause License; see [LICENSE](https://github.com/timothygebhard/hsr4hci/blob/master/LICENSE) for more details.
