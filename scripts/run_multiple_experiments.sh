#!/bin/bash
# Run multiple experiments with different configurations
python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_new_sampling
python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_cheaper_eps_suboptimal_sampling
python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_cheaper_optimal_sampling
python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_new_containment
python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_cheaper_optimal_containment
python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name reaches_cheaper_eps_suboptimal_containment
# Baseline method
python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name ixg_baseline
# python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name ixg_star_eps_suboptimal_baseline
# python3 experiments/run_contact_graph_experiment.py --config-path ../config/WAFR_experiments --config-dir config --config-name ixg_star_optimal_baseline



# Full Graph Creation
# python3 scripts/full_contact_graph_generator.py cg_simple_3
# python3 scripts/full_contact_graph_generator.py cg_trichal4
# python3 scripts/full_contact_graph_generator.py cg_maze_b1

# GGCS
# python3 experiments/run_ggcs_se3_maze_experiment.py --config-path ../config/ggcs_experiments --config-dir config --config-name reaches_cheaper_eps_suboptimal_sampling_ggcs
# python3 experiments/run_ggcs_se3_maze_experiment.py --config-path ../config/ggcs_experiments --config-dir config --config-name reaches_new_sampling_ggcs
# python3 experiments/run_ggcs_se3_maze_experiment.py --config-path ../config/ggcs_experiments --config-dir config --config-name reaches_cheaper_optimal_sampling_ggcs

# Naive Astar on GCS
# python3 experiments/run_contact_graph_experiment.py --config-name cg_gcs_naive_astar