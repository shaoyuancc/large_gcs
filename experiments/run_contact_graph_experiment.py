import importlib
import logging
import os
from dataclasses import asdict
from datetime import datetime

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import get_original_cwd, instantiate
from omegaconf import OmegaConf, open_dict

import wandb
from large_gcs.graph.contact_graph import ContactGraph
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

    graph_file = ContactGraphGeneratorParams.graph_file_path_from_name(cfg.graph_name)
    cg = ContactGraph.load_from_file(graph_file)

    module_name, function_name = cfg.contact_shortcut_edge_cost_factory.rsplit(".", 1)
    module = importlib.import_module(module_name)
    shortcut_cost = getattr(module, function_name)
    alg = instantiate(cfg.algorithm, graph=cg, shortcut_edge_cost_factory=shortcut_cost)
    sol = alg.run(animate=False)

    if cfg.save_visualization:
        output_base = f"{alg.__class__.__name__}_{function_name}_{cfg.graph_name}"
        vid_file = os.path.join(full_log_dir, f"{output_base}.mp4")
        graphviz_file = os.path.join(full_log_dir, f"{output_base}_visited_subgraph")
        gviz = alg._visited.graphviz()
        gviz.format = "pdf"
        gviz.render(graphviz_file, view=False)

        anim = cg.animate_solution()
        anim.save(vid_file)
        if cfg.save_to_wandb:
            wandb.log({"animation": wandb.Video(vid_file)})
            wandb.save(graphviz_file + ".pdf")

    logger.info(f"hydra log dir: {full_log_dir}")

    if cfg.save_to_wandb:
        wandb.run.summary["final_sol"] = sol.to_serializable_dict()
        wandb.run.summary["alg_metrics"] = asdict(alg.alg_metrics)

        wandb.finish()


if __name__ == "__main__":
    main()
