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

## Runing a single experiment

Create a config file specifying the experiment in `config` and run it using the following command:

```
python3 experiments/run_contact_graph_experiment.py --config-name cg_gcs_astar_conv_res
```

where `cg_gcs_astar_conv_res` should be replaced with your config name.

## Running multiple experiments

Create a bash script in `scripts` and make it executable with `chmod +x run_multiple_experiments.sh`.
Then run it with `run_multiple_experiments.sh`

## Running tests

In typical development, regularly run
`pytest -m "not slow_test"`
which runs all the tests that take shorter than 5 seconds each.

To run the slow tests alone use
`pytest -m slow_test`

Or to run all the tests simply use
`pytest`

To run a specific test(s) by referring to some part of the test's name, use
`pytest -k "shortcut_edge_cg_simple_2_inc"`

To make tests verbose and allow print statements to be shown use `-v -s` flags.

## Credits
This repo references and contains code from: Bernhard Paus Græsdal https://github.com/bernhardpg/planning-through-contact and Tobia Marcucci https://github.com/TobiaMarcucci/shortest-paths-in-graphs-of-convex-sets.