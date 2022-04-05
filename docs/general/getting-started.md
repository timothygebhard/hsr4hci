# Getting started

This short guide will walk you through the required steps to set up and install `hsr4hci`.


## Installing the hsr4hci package

The code in this repository is organized as a Python package named `hsr4hci` together with a set of scripts that use the functions and classes of the package.
To get started, you will need to install the package.
For this, we *strongly* recommend you to use a [virtual environment](https://virtualenv.pypa.io/en/latest/). 

```{attention} 
The code was written for **Python 3.8 and above**; earlier versions will likely require some small modifications.
``` 

Once you have set up a suitable virtualenv, clone this repository and install `hsr4hci` as a Python package:

```bash
git clone git@github.com:timothygebhard/hsr4hci.git ;
cd hsr4hci ;
pip install .
```

In case you intend to not only use, but also develop or modify things about the package, it may be useful to install the package in "edit mode" by using the `-e` flag:

```bash
pip install -e .
```

In this case, you might also want to install the "development requirements":

```bash
pip install -e ".[develop]"
```

This will install, for example, `pytest` to run the unit tests.


## Setting up the required environmental variables

The workflows of this code base rely on two environmental variables that you need to set to let the code know where the `datasets` and the `experiments` folder are located.
To this end, please use:

```bash
export HSR4HCI_DATASETS_DIR="/path/to/datasets/dir" ;
export HSR4HCI_EXPERIMENTS_DIR="/path/to/experiments/dir" ;
```

To make these changes more permanent, you can add these lines to your `.bashrc`, `.zshrc`, ...

*Background:* The `datasets` and the `experiments` folder are contained at the top level of the `hsr4hci` repository.
However, if you do not install the package in edit mode, the code has no way of knowing where you placed these folders when you cloned the repository :)
Additionally, you might also want to move them to a different location or disk than the code itself (e.g., due to storage reasons).
