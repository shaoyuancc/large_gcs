#!/bin/bash
# Run multiple experiments with different configurations
python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_new_sampling
python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_cheaper_eps_suboptimal_sampling
python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_cheaper_optimal_sampling
python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_new_containment
python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_cheaper_eps_suboptimal_containment
python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_cheaper_optimal_containment
