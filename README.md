# large_gcs
Combining ideas from graph search and graph of convex sets

## Installation (Linux and MacOS)
This repo uses Poetry for dependency management. To setup this project, first install [Poetry](https://python-poetry.org/docs/#installation) and, make sure to have Python3.10 installed on your system.

Then, configure poetry to setup a virtual environment that uses >= Python 3.8:
```
poetry env use python3.11
```

Next, install all the required dependencies to the virtual environment with the following command:
```
poetry install -vvv
```
(the `-vvv` flag adds verbose output).

Next, add the following lines to `.venv/bin/activate` replacing the paths with your actual paths:
```
export PROJECT_ROOT="/path/to/project/folder/large_gcs"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export MOSEKLM_LICENSE_FILE="/path/to/mosek/license/mosek/mosek.lic"
export GRB_LICENSE_FILE="/path/to/gurobi/license/gurobi.lic"
```

Finally, make sure to have graphviz installed on your computer. On MacOS, run the following command:
```
brew install graphviz
```

## Running pre-commit hooks
The repo is setup to do automatic linting and code checking on every commit through the use of pre-commits. To run all the pre-commit hooks (which will clean up all files in the repo), run the following command:
```
poetry shell
pre-commit install
```