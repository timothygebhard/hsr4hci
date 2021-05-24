# hsr4hci: Half-Sibling Regression for High-Contrast Imaging

![Python 3.7](https://img.shields.io/badge/python-v3.7-blue)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Checked with MyPy](https://img.shields.io/badge/mypy-checked-blue)](https://github.com/python/mypy)
![Tests](https://github.com/timothygebhard/hsr4hci/workflows/Tests/badge.svg?branch=master)
![Coverage Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/timothygebhard/40d8bf48dcbaf33c99e8de35ad6161f2/raw/hsr4hci.json)


## ⚡ Quickstart

To get started, clone this repository and install `hsr4hci` as a Python package:

```
git clone git@github.com:timothygebhard/hsr4hci.git
cd hsr4hci
pip install .
```

If you want to use "developer options" (e.g., run unit tests), change the last line to:

```
pip install .[develop]
```

To run any experiments or reproduce our results, you will first need to create some data sets in the right format.
Please check out the [README file in the `datasets` directory](https://github.com/timothygebhard/hsr4hci/tree/master/datasets) for more information on how to do this.
