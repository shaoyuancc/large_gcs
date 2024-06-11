#!/bin/bash

TIMEOUT_DURATION=10h

# Note: on Mac you'll need to install coreutils to get timeout `brew install coreutils`

# Run multiple experiments with different configurations
timeout $TIMEOUT_DURATION python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_new_sampling
timeout $TIMEOUT_DURATION python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_cheaper_eps_suboptimal_sampling
timeout $TIMEOUT_DURATION python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_cheaper_optimal_sampling
timeout $TIMEOUT_DURATION python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_new_containment
timeout $TIMEOUT_DURATION python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_cheaper_optimal_containment
# timeout $TIMEOUT_DURATION python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_cheaper_eps_suboptimal_containment
# Baseline method
timeout $TIMEOUT_DURATION python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name ixg_baseline
timeout $TIMEOUT_DURATION python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name ixg_star_eps_suboptimal_baseline
# python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name ixg_star_optimal_baseline

# Check the exit status
if [ $? -eq 124 ]; then
  echo "The script timed out after $TIMEOUT_DURATION"
else
  echo "The script completed."
fi
