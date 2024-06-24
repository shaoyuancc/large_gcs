import argparse
import logging
import os
import time

import numpy as np

from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.lower_bound_graph import LowerBoundGraph
from large_gcs.graph_generators.contact_graph_generator import (
    ContactGraphGeneratorParams,
)

logger = logging.getLogger(__name__)

GRAPHS = ["cg_simple_1_1", "cg_simple_1", "cg_simple_4"]


def generate_lbg(graph_name: str, checkpoint: bool):
    graph_file = ContactGraphGeneratorParams.graph_file_path_from_name(graph_name)
    cg = ContactGraph.load_from_file(
        graph_file,
        should_use_l1_norm_vertex_cost=True,
    )
    # cg.plot()
    lbg = LowerBoundGraph.generate_from_gcs(
        graph_name, cg, save_to_file=True, start_from_checkpoint=checkpoint
    )


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    parser = argparse.ArgumentParser(
        description="Generate lower bound graph for IxG and IxG*"
    )
    parser.add_argument(
        "graph_names",
        type=str,
        nargs="+",  # This allows one or more graph names to be passed
        help="Name(s) of the graph(s) to generate, or 'all' to generate all graphs",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        action="store_true",
        help="Continue from checkpoint",
    )
    args = parser.parse_args()
    # Create log directory relative to the script location
    log_dir = os.path.join(os.environ["PROJECT_ROOT"], "output")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    graph_name_str = "_".join(args.graph_names)
    graph_name_str = graph_name_str.replace(" ", "_")

    current_date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(
        log_dir, f"{current_date_time}_{graph_name_str}_lbg.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    )
    logging.getLogger("large_gcs").setLevel(logging.DEBUG)
    logging.getLogger("drake").setLevel(logging.WARNING)
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    if "all" in args.graph_names:
        graph_names_to_generate = GRAPHS
    else:
        graph_names_to_generate = args.graph_names

    for graph_name in graph_names_to_generate:
        generate_lbg(graph_name, args.checkpoint)

    logger.info(f"log saved to {log_file_path}")
