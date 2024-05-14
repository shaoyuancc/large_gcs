import argparse
import logging
from pathlib import Path

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

logger = logging.getLogger(__name__)


# TODO: This code is from run_contact_graph_experiment and could be merged.
def _construct_graph(cfg: DictConfig) -> ContactGraph:
    if cfg.should_use_incremental_graph:
        graph_file = ContactGraphGeneratorParams.inc_graph_file_path_from_name(
            cfg.graph_name
        )
        cg = IncrementalContactGraph.load_from_file(
            graph_file,
            should_incl_simul_mode_switches=cfg.should_incl_simul_mode_switches,
            should_add_const_edge_cost=cfg.should_add_const_edge_cost,
            should_add_gcs=(
                True
                if (
                    "abstraction_model_generator" in cfg
                    or cfg.algorithm._target_
                    == "large_gcs.algorithms.gcs_astar_reachability.GcsAstarReachability"
                )
                else False
            ),
        )
    else:
        graph_file = ContactGraphGeneratorParams.graph_file_path_from_name(
            cfg.graph_name
        )
        cg = ContactGraph.load_from_file(graph_file)

    return cg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load data and configuration from a specified directory"
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
        cg = _construct_graph(cfg)

        sol_files = list(path.glob("*_solution.pkl"))
        if not len(sol_files) == 1:
            raise RuntimeError(
                f"Found more than one solution file in {path}."
                f"This is not expected, so something is likely wrong."
            )
        sol_file = sol_files[0]
        sol = ShortestPathSolution.load(sol_file)

        # Generate all required neighbours in graph
        cg.set_target("target")
        for v in sol.vertex_path:
            if v == "target":
                continue
            cg.generate_neighbors(v)

        metric_files = list(path.glob("*_metrics.json"))
        if not len(metric_files) == 1:
            raise RuntimeError(
                f"Found more than one metric file in {path}."
                f"This is not expected, so something is likely wrong."
            )
        metric_file = metric_files[0]
        metrics = AlgMetrics.load(metric_file)

        cg.contact_spp_sol = cg.create_contact_spp_sol(
            sol.vertex_path, sol.ambient_path
        )
        vid_file = path / "regenerated_video.mp4"

        anim = cg.animate_solution()
        anim.save(vid_file)

        cg.plot_solution(cg.contact_spp_sol, path / "traj_figure.pdf")


if __name__ == "__main__":
    main()
