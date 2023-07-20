import hydra
from hydra.utils import instantiate, call, get_original_cwd
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf, open_dict
import wandb
import importlib
import logging
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.example_graphs.utils.contact_graph_generator import (
    ContactGraphGeneratorParams,
)


@hydra.main(version_base=None, config_path="../config", config_name="basic")
def main(cfg: OmegaConf) -> None:
    graph_file = ContactGraphGeneratorParams.graph_file_path_from_name(cfg.graph_name)
    cg = ContactGraph.load_from_file(graph_file)

    module_name, function_name = cfg.contact_shortcut_edge_cost_factory.rsplit(".", 1)
    module = importlib.import_module(module_name)
    shortcut_cost = getattr(module, function_name)
    alg = instantiate(cfg.algorithm, graph=cg, shortcut_edge_cost_factory=shortcut_cost)
    sol = alg.run(animate=False)


if __name__ == "__main__":
    main()
