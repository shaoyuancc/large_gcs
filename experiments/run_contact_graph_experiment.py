import logging
import os
from datetime import datetime

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import get_original_cwd, instantiate
from omegaconf import OmegaConf, open_dict

import wandb
from large_gcs.algorithms.gcs_hastar import GcsHAstar
from large_gcs.algorithms.search_algorithm import SearchAlgorithm
from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.graph import ShortestPathSolution
from large_gcs.graph.incremental_contact_graph import IncrementalContactGraph
from large_gcs.graph_generators.contact_graph_generator import (
    ContactGraphGeneratorParams,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="basic")
def main(cfg: OmegaConf) -> None:
    now = datetime.now()
    now.strftime("%Y-%m-%d_%H-%M-%S")
    # Add log dir to config
    hydra_config = HydraConfig.get()
    full_log_dir = hydra_config.runtime.output_dir
    with open_dict(cfg):
        cfg.log_dir = os.path.relpath(full_log_dir, get_original_cwd() + "/outputs")

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
                name=cfg.log_dir,
                config=wandb_config,
                group=hydra_config.job.config_name + "_" + folder_name,
                save_code=True,
            )
        else:
            wandb.init(
                project="large_gcs",
                name=cfg.log_dir,
                config=wandb_config,
                save_code=True,
            )
        wandb.run.log_code(root=os.path.join(os.environ["PROJECT_ROOT"], "large_gcs"))

    logger.info(cfg)
    if cfg.should_use_incremental_graph:
        graph_file = ContactGraphGeneratorParams.inc_graph_file_path_from_name(
            cfg.graph_name
        )
        cg = IncrementalContactGraph.load_from_file(
            graph_file,
            should_incl_simul_mode_switches=cfg.should_incl_simul_mode_switches,
            should_add_const_edge_cost=cfg.should_add_const_edge_cost,
            should_add_gcs=(True if "abstraction_model_generator" in cfg else False),
        )
    else:
        graph_file = ContactGraphGeneratorParams.graph_file_path_from_name(
            cfg.graph_name
        )
        cg = ContactGraph.load_from_file(graph_file)
    if "abstraction_model_generator" in cfg:
        print("Using abstraction model generator")
        abs_model_generator = instantiate(cfg.abstraction_model_generator)
        abs_model = abs_model_generator.generate(concrete_graph=cg)
        alg = GcsHAstar(
            abs_model=abs_model, reexplore_levels=cfg.algorithm.reexplore_levels
        )
    else:
        print("No abstraction model generator")
        cost_estimator: CostEstimator = instantiate(cfg.cost_estimator, graph=cg)
        alg: SearchAlgorithm = instantiate(
            cfg.algorithm, graph=cg, cost_estimator=cost_estimator
        )

    sol: ShortestPathSolution = alg.run()

    if sol is not None and cfg.save_visualization:
        if "abstraction_model_generator" in cfg:
            model_name = cfg.abstraction_model_generator["_target_"].split(".")[-1]
            output_base = f"{alg.__class__.__name__}_{model_name}_{cfg.graph_name}"
        else:
            output_base = f"{alg.__class__.__name__}_{cost_estimator.finger_print}_{cfg.graph_name}"
        vid_file = os.path.join(full_log_dir, f"{output_base}.mp4")
        # graphviz_file = os.path.join(full_log_dir, f"{output_base}_visited_subgraph")
        # gviz = alg._visited.graphviz()
        # gviz.format = "pdf"
        # gviz.render(graphviz_file, view=False)

        anim = cg.animate_solution()
        anim.save(vid_file)
        if cfg.save_to_wandb:
            wandb.log({"animation": wandb.Video(vid_file)})
            # wandb.save(graphviz_file + ".pdf")

    logger.info(f"hydra log dir: {full_log_dir}")

    if cfg.save_to_wandb:
        if sol is not None:
            wandb.run.summary["final_sol"] = sol.to_serializable_dict()
        wandb.run.summary["alg_metrics"] = alg.alg_metrics.to_dict()

        wandb.finish()


if __name__ == "__main__":
    main()
