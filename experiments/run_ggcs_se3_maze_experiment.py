import logging
import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import get_original_cwd, instantiate
from omegaconf import OmegaConf, open_dict

import wandb
from large_gcs.algorithms.gcs_hastar import GcsHAstar
from large_gcs.algorithms.gcs_hastar_reachability import GcsHAstarReachability
from large_gcs.algorithms.search_algorithm import AlgVisParams, SearchAlgorithm
from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.domination_checkers.domination_checker import DominationChecker
from large_gcs.graph.cfree_graph import CFreeGraph
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.graph import ShortestPathSolution
from large_gcs.graph.incremental_contact_graph import IncrementalContactGraph
from large_gcs.graph.lower_bound_graph import LowerBoundGraph
from large_gcs.graph_generators.contact_graph_generator import (
    ContactGraphGeneratorParams,
)
from large_gcs.utils.hydra_utils import get_cfg_from_folder

logger = logging.getLogger(__name__)

BASELINE_ALGS = [
    "large_gcs.algorithms.ixg.IxG",
    "large_gcs.algorithms.ixg_star.IxGStar",
]


@hydra.main(version_base=None, config_path="../config", config_name="basic")
def main(cfg: OmegaConf) -> None:
    # Add log dir to config
    hydra_config = HydraConfig.get()
    full_log_dir = hydra_config.runtime.output_dir
    with open_dict(cfg):
        cfg.log_dir = os.path.relpath(full_log_dir, get_original_cwd() + "/outputs")

    # Save the configuration to the log directory
    run_folder = Path(full_log_dir)
    config_file = run_folder / "config.yaml"
    with open(config_file, "w") as f:
        OmegaConf.save(cfg, f)

    if cfg.save_to_wandb:
        wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb_config["log_dir"] = cfg.log_dir
        if hydra_config.mode == RunMode.MULTIRUN:
            # Extract folder name from sweep dir
            folder_name = folder_name = [
                x for x in hydra_config.sweep.dir.split("/") if x
            ][-1]
            wandb.init(
                project="large_gcs",
                entity="contact_placement",
                name=cfg.log_dir,
                config=wandb_config,
                group=hydra_config.job.config_name + "_" + folder_name,
                save_code=True,
            )
        else:
            wandb.init(
                project="large_gcs",
                entity="contact_placement",
                name=cfg.log_dir,
                config=wandb_config,
                save_code=True,
            )
        wandb.run.log_code(root=os.path.join(os.environ["PROJECT_ROOT"], "large_gcs"))

    logger.info(cfg)
    g = CFreeGraph.load_from_file(cfg.graph_name)

    cost_estimator: CostEstimator = instantiate(
        cfg.cost_estimator, graph=g, add_const_cost=False
    )
    domination_checker: DominationChecker = instantiate(cfg.domination_checker, graph=g)
    if cfg.algorithm._target_ in BASELINE_ALGS:
        raise NotImplementedError("Baseline algorithms not supported in this script.")

    alg: SearchAlgorithm = instantiate(
        cfg.algorithm,
        graph=g,
        cost_estimator=cost_estimator,
        domination_checker=domination_checker,
        vis_params=AlgVisParams(log_dir=full_log_dir),
    )

    sol: ShortestPathSolution = alg.run()

    save_outputs = cfg.save_metrics or cfg.save_visualization or cfg.save_solution
    if save_outputs:
        if cfg.algorithm._target_ in BASELINE_ALGS:
            output_base = f"{alg.__class__.__name__}_{cfg.graph_name}"
        else:
            output_base = (
                f"{alg.__class__.__name__}_"
                + f"{cost_estimator.finger_print}_{cfg.graph_name}"
            )

    if sol is not None and cfg.save_metrics:
        metrics_path = Path(full_log_dir) / f"{output_base}metrics.json"
        alg.save_alg_metrics(metrics_path)

        if cfg.save_to_wandb:
            wandb.save(str(metrics_path))  # type: ignore

    if sol is not None and cfg.save_solution:
        sol_path = Path(full_log_dir) / f"{output_base}solution.pkl"
        sol.save(sol_path)

        if cfg.save_to_wandb:
            wandb.save(str(sol_path))  # type: ignore

    logger.info(f"hydra log dir: {full_log_dir}")

    if cfg.save_to_wandb:
        if sol is not None:
            wandb.run.summary["final_sol"] = sol.to_serializable_dict()
        wandb.run.summary["alg_metrics"] = alg.alg_metrics.to_dict()

        wandb.finish()


if __name__ == "__main__":
    main()
