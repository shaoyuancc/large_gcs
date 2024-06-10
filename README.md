# large_gcs
Combining ideas from graph search and graph of convex sets

## Installation (Linux and MacOS)
This repo uses Poetry for dependency management. To setup this project, first install [Poetry](https://python-poetry.org/docs/#installation) and, make sure to have Python3.12 installed on your system.

(Optional) Configure poetry to create virtual environment in project
```
poetry config virtualenvs.in-project true
```

Then, configure poetry to setup a virtual environment that uses >= Python 3.12:
```
poetry env use python3.12
```

Next, install all the required dependencies to the virtual environment with the following command:
```
poetry install -vvv
```
(the `-vvv` flag adds verbose output).

Clone this fork of pypolycontain locally.
https://github.com/shaoyuancc/pypolycontain

Next, add the following lines to `.venv/bin/activate` replacing the paths with your actual paths:
```
PYPOLYCONTAIN="/path/to/pypolycontain"
export PROJECT_ROOT="/path/to/project/folder/large_gcs"
export PYTHONPATH="$PROJECT_ROOT:$PYPOLYCONTAIN:$PYTHONPATH"
export MOSEKLM_LICENSE_FILE="/path/to/mosek/license/mosek/mosek.lic"
export GRB_LICENSE_FILE="/path/to/gurobi/license/gurobi.lic"
```
### Additional libraries used
On Ubuntu
```
sudo apt install graphviz
sudo apt-get install python3-tk # For interactive plotting in graph generator script
sudo apt install ffmpeg # For saving animation videos
```
On MacOS, run the following command:
```
brew install graphviz
brew install python-tk 
brew install ffmpeg
```

Additional packages that I was not able to install using poetry, and had to use pip instead
```
pip install kaleido
```

If you have problems with poetry, when in the virtual env shell, run the following:
```
pip install numpy matplotlib ipykernel scipy graphviz black tqdm pytest wandb hydra-core omegaconf autoflake isort pdbpp plotly docformatter drake kaleido pre-commit
```

## Running pre-commit hooks
The repo is setup to do automatic linting and code checking on every commit through the use of pre-commits. To run all the pre-commit hooks (which will clean up all files in the repo), run the following command:
```
poetry shell
pre-commit install
```

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

## Runing a single experiment

Create a config file specifying the experiment in `config` and run it using the following command:

```
python3 experiments/run_contact_graph_experiment.py --config-name quickstart
```

where `quickstart` should be replaced with your config name.

After running quickstart you can compare your results (which should appear in the `multirun/date/time` folder) to the contents of the `quickstart_output` folder

## Runing WAFR experiments
```
python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name trajectory_figures 
```

## Running multiple experiments

Create a bash script in `scripts` and make it executable with `chmod +x run_multiple_experiments.sh`.
Then run it with `run_multiple_experiments.sh`

## Generating figures
On Ubuntu, 'Times' font can be installed via
```
sudo apt update
sudo apt install ttf-mscorefonts-installer
```
After installation, you may need to update the font cache:

```
fc-cache -f -v
```

On Mac, it already comes in the system

Additionally, need LaTex on your system
For Ubuntu,
```
sudo apt install texlive-latex-extra
sudo apt-get install texlive-fonts-recommended
sudo apt-get install cm-super
sudo apt-get install dvipng
```

## Credits
This repo references and contains code from: Bernhard Paus Græsdal https://github.com/bernhardpg/planning-through-contact and Tobia Marcucci https://github.com/TobiaMarcucci/shortest-paths-in-graphs-of-convex-sets.

