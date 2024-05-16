import argparse
from pathlib import Path
from typing import NamedTuple

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

import wandb
from large_gcs.algorithms.gcs_hastar import GcsHAstar
from large_gcs.algorithms.gcs_hastar_reachability import GcsHAstarReachability
from large_gcs.algorithms.search_algorithm import (
    AlgMetrics,
    AlgVisParams,
    SearchAlgorithm,
)
from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.domination_checkers.domination_checker import DominationChecker
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.graph import ShortestPathSolution
from large_gcs.graph.incremental_contact_graph import IncrementalContactGraph
from large_gcs.graph_generators.contact_graph_generator import (
    ContactGraphGeneratorParams,
)


class SingleRunData(NamedTuple):
    num_samples: int
    cost: float
    wall_clock_time: float
    num_paths_expanded: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load data and configuration from a specified directory."
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Path to the directory containing data and config file",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Get the directory path
    data_dir = Path(args.dir)

    # Validate the directory
    if not data_dir.is_dir():
        print(f"The directory {data_dir} does not exist.")
        return

    run_data = []

    # Iterate through the runs and generate figures
    for path in data_dir.iterdir():
        # We only care about the subfolders (which contain data for a run)
        if not path.is_dir():
            continue

        # Load the config file for this run
        config_path = path / "config.yaml"
        if not config_path.is_file():
            print(f"The config file {config_path} does not exist.")
            return
        cfg: DictConfig = OmegaConf.load(config_path)  # type: ignore

        sol_files = list(path.glob("*_solution.pkl"))
        if not len(sol_files) == 1:
            raise RuntimeError(
                f"Found more than one solution file in {path}."
                f"This is not expected, so something is likely wrong."
            )
        sol_file = sol_files[0]
        sol = ShortestPathSolution.load(sol_file)

        metric_files = list(path.glob("*_metrics.json"))
        if not len(metric_files) == 1:
            raise RuntimeError(
                f"Found more than one metric file in {path}."
                f"This is not expected, so something is likely wrong."
            )
        metric_file = metric_files[0]
        metrics = AlgMetrics.load(metric_file)

        num_samples = cfg.domination_checker.num_samples_per_vertex

        data = SingleRunData(num_samples, sol.cost, metrics.time_wall_clock, 0)
        run_data.append(data)

    breakpoint()


if __name__ == "__main__":
    main()
