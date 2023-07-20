from dataclasses import asdict
import hydra
from hydra.utils import instantiate, call, get_original_cwd
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf, open_dict
import wandb
import importlib
import logging
from datetime import datetime
import os
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.example_graphs.utils.contact_graph_generator import (
    ContactGraphGeneratorParams,
)


@hydra.main(version_base=None, config_path="../config", config_name="basic")
def main(cfg: OmegaConf) -> None:
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
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
            )
        else:
            wandb.init(
                project="large_gcs",
                name=cfg.log_dir,
                config=wandb_config,
            )
    logging.info(cfg)

    graph_file = ContactGraphGeneratorParams.graph_file_path_from_name(cfg.graph_name)
    cg = ContactGraph.load_from_file(graph_file)

    module_name, function_name = cfg.contact_shortcut_edge_cost_factory.rsplit(".", 1)
    module = importlib.import_module(module_name)
    shortcut_cost = getattr(module, function_name)
    alg = instantiate(cfg.algorithm, graph=cg, shortcut_edge_cost_factory=shortcut_cost)
    sol = alg.run(animate=False)

    # vid_file = os.path.join(full_log_dir, f"{method_modifier}_{base_filename}.mp4")
    # graphviz_file = os.path.join(output_dir, f"{method_modifier}_{base_filename}")
    # gviz = gcs_astar._visited.graphviz()
    # gviz.format = "pdf"
    # gviz.render(graphviz_file, view=False)

    # anim = cg.animate_solution()
    # anim.save(vid_file)

    logging.info(f"hydra log dir: {full_log_dir}")

    if cfg.save_to_wandb:
        wandb.run.summary["final_sol"] = sol.to_serializable_dict()
        wandb.run.summary["alg_metrics"] = asdict(alg.alg_metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
