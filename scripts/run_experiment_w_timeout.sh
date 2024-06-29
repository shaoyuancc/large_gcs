#!/bin/bash

TIMEOUT_DURATION=10h
LOG_FILE="multirun/experiment_log.txt"

# Function to run experiment and log result
run_experiment() {
    timeout $TIMEOUT_DURATION python3 experiments/run_contact_graph_experiment.py "$@"
    if [ $? -eq 124 ]; then
        echo "$(date) - Experiment $@ timed out after $TIMEOUT_DURATION" | tee -a "$LOG_FILE"
    else
        echo "$(date) - Experiment $@ completed" | tee -a "$LOG_FILE"
    fi
}

# Run multiple experiments with different configurations
# run_experiment --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_new_sampling
# run_experiment --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_cheaper_eps_suboptimal_sampling

# run_experiment --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_new_containment
run_experiment --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_cheaper_eps_suboptimal_containment
run_experiment --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_cheaper_optimal_containment
run_experiment --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_cheaper_optimal_sampling

# Baseline method
# run_experiment --config-path ../config/WAFR_experiments --config-dir config --config-name ixg_baseline
run_experiment --config-path ../config/WAFR_experiments --config-dir config --config-name ixg_star_eps_suboptimal_baseline
run_experiment --config-path ../config/WAFR_experiments --config-dir config --config-name ixg_star_optimal_baseline

# Check if any experiment timed out
if grep -q "timed out" "$LOG_FILE"; then
  echo "run_experiment_w_timeout complete. At least one script timed out. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
else
  echo "run_experiment_w_timeout complete. All scripts completed without timing out. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
fi
