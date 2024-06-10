import argparse
import logging
import os
import time

import numpy as np
from matplotlib import pyplot as plt

from large_gcs.contact.contact_regions_set import ContactRegionParams
from large_gcs.graph_generators.contact_graph_generator import (
    ContactGraphGenerator,
    ContactGraphGeneratorParams,
)

# Ensure the backend is set for interactive plotting
# plt.switch_backend('TkAgg')

logger = logging.getLogger(__name__)


def generate_graph(graph_name: str, incremental, preview):
    params = GRAPH_PARAMS.get(graph_name)
    if not params:
        raise ValueError(f"Unknown graph name: {graph_name}")

    generator = ContactGraphGenerator(params)
    if preview:
        raise NotImplementedError("Previewing graphs is not yet implemented")
        generator.plot()
        plt.title(params.name)
        plt.show()
        return

    start_time = time.time()
    logger.info(f"Starting graph creation of {params.name}")

    if incremental:
        graph = generator.generate_incremental_contact_graph()
    else:
        graph = generator.generate()
    end_time = time.time()
    logger.info(f"Graph creation took: {end_time - start_time} seconds")
    logger.info(
        f"duration in H:M:S {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}"
    )
    logger.info(graph.params)


GRAPH_PARAMS = {
    "cg_simple_1_1": ContactGraphGeneratorParams(
        name="cg_simple_1_1",
        obs_vertices=[np.array([[0, 0], [2, 0], [2, 2], [0, 2]]) + np.array([-1, -1])],
        obj_vertices=[],
        rob_vertices=[np.array([[0, 0], [1, 0], [1, 1], [0, 1]])],
        source_obj_pos=[],
        source_rob_pos=[[-2, -2]],
        target_region_params=[
            ContactRegionParams(
                region_vertices=[[1.5, -1], [1.5, 1], [3, 1], [3, -1]], rob_indices=[0]
            ),
        ],
        n_pos_per_set=2,
        workspace=[[-3, 3], [-3, 3]],
        should_add_const_edge_cost=True,
        should_use_l1_norm_vertex_cost=True,
    ),
    "cg_simple_1": ContactGraphGeneratorParams(
        name="cg_simple_1",
        obs_vertices=[],
        obj_vertices=[np.array([[0, 0], [1, 0], [1, 1], [0, 1]])],
        rob_vertices=[np.array([[0, 0], [1, 0], [1, 1], [0, 1]])],
        source_obj_pos=[[0, 0]],
        source_rob_pos=[[-2, -2]],
        target_region_params=[
            ContactRegionParams(
                region_vertices=[[1.5, -1], [1.5, 1], [3, 1], [3, -1]], obj_indices=[0]
            ),
        ],
        n_pos_per_set=2,
        workspace=[[-3, 3], [-3, 3]],
        should_use_l1_norm_vertex_cost=True,
    ),
    "cg_simple_3": ContactGraphGeneratorParams(
        name="cg_simple_3",
        obs_vertices=[],
        obj_vertices=[
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) + np.array([2.5, 0.5])
        ],
        rob_vertices=[np.array([[-1, -1], [-1.5, -0.5], [-1.2, -1.5]])],
        source_obj_pos=[[0, 0]],
        source_rob_pos=[[-2, -2]],
        target_region_params=[
            ContactRegionParams(
                region_vertices=[[1.5, -1], [1.5, 1], [3, 1], [3, -1]], obj_indices=[0]
            ),
        ],
        n_pos_per_set=2,
        workspace=[[-3, 3], [-3, 3]],
        should_add_const_edge_cost=True,
        should_use_l1_norm_vertex_cost=True,
    ),
    "cg_simple_4": ContactGraphGeneratorParams(
        name="cg_simple_4",
        obs_vertices=[np.array([[0, 0], [2, 0], [2, 2], [0, 2]]) + np.array([-1, -1])],
        obj_vertices=[np.array([[0, 0], [1, 0], [1, 1], [0, 1]])],
        rob_vertices=[np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]])],
        source_obj_pos=[[-2, 0]],
        source_rob_pos=[[0, -2]],
        target_region_params=[
            ContactRegionParams(
                region_vertices=[[1.5, -1], [1.5, 1], [3, 1], [3, -1]], obj_indices=[0]
            ),
        ],
        n_pos_per_set=2,
        workspace=[[-3.5, 3.5], [-3.5, 3.5]],
        should_add_const_edge_cost=True,
        should_use_l1_norm_vertex_cost=True,
    ),
    "cg_maze_b1": ContactGraphGeneratorParams(
        name="cg_maze_b1",
        obs_vertices=[
            [[-2.360, -3.821], [1.840, 1.479], [-1.660, -0.621]],
        ],
        obj_vertices=[
            [[1.000, 0.500], [1.000, -0.500], [2.000, -0.500], [2.000, 0.500]]
        ],
        rob_vertices=[[[3.000, 1.000], [3.000, 0.000], [3.500, 0.000]]],
        source_obj_pos=[[0.329, -2.245]],
        source_rob_pos=[[-0.037, 2.527]],
        n_pos_per_set=2,
        workspace=[[-5, 5], [-5, 5]],
        target_region_params=[
            ContactRegionParams(
                region_vertices=[
                    [-3.867, -2.508],
                    [-2.767, -2.508],
                    [-2.767, -1.408],
                    [-3.867, -1.408],
                ],
                obj_indices=[0],
                rob_indices=None,
            )
        ],
        should_add_const_edge_cost=True,
        should_use_l1_norm_vertex_cost=True,
    ),
    "cg_trichal4": ContactGraphGeneratorParams(
        name="cg_trichal4",
        obs_vertices=[[[-1, 2], [-1, -1], [2, 2]]],
        obj_vertices=[[[1, 0.5], [1, -0.5], [2, -0.5], [2, 0.5]]],
        rob_vertices=[[[3, 1], [3, 0], [3.5, 0]]],
        source_obj_pos=[[3.25, 0]],
        source_rob_pos=[[1.5, 0.5]],
        target_region_params=[
            ContactRegionParams(
                region_vertices=[[-3, -0.75], [-3, 1], [-1, -0.75], [-1, 1]],
                obj_indices=[0],
            ),
        ],
        n_pos_per_set=2,
        workspace=[[-5, 5], [-4, 4]],
        should_add_const_edge_cost=True,
        should_use_l1_norm_vertex_cost=True,
    ),
    "cg_stackpush_d2": ContactGraphGeneratorParams(
        name="cg_stackpush_d2",
        obs_vertices=[[[-3, -2], [2.2, 1], [3, 1], [3, -2]]],
        obj_vertices=[
            [[1, 0.5], [1, -0.5], [2, -0.5], [2, 0.5]],
            [[1, 0.5], [1, -0.5], [2, -0.5], [2, 0.5]],
            [[1, 0.5], [1, -0.5], [2, -0.5], [2, 0.5]],
        ],
        rob_vertices=[[[0, 0], [0, 0.5], [1.5, 0], [1.5, 0.5]]],
        source_obj_pos=[[-2.5, -0.7], [-1.1, 0.1], [0.4, 1]],
        source_rob_pos=[[-5, 1]] + np.array([2, -4]),
        target_region_params=[
            ContactRegionParams(
                region_vertices=[[3, 0], [3, -3], [6, 0], [6, -3]],
                obj_indices=[0, 1, 2],
            ),
        ],
        n_pos_per_set=2,
        workspace=[[-7, 7], [-5, 5]],
        should_add_const_edge_cost=True,
        should_use_l1_norm_vertex_cost=True,
    ),
}


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    parser = argparse.ArgumentParser(description="Generate contact graph")
    parser.add_argument(
        "graph_names",
        type=str,
        nargs="+",  # This allows one or more graph names to be passed
        help="Name(s) of the graph(s) to generate, or 'all' to generate all graphs",
    )
    parser.add_argument(
        "-i", "--incremental", action="store_true", help="Generate incremental graphs"
    )
    parser.add_argument(
        "-p",
        "--preview",
        action="store_true",
        help="Plot the graphs and do not generate",
    )
    args = parser.parse_args()
    if not args.preview:
        # Create log directory relative to the script location
        log_dir = os.path.join(os.environ["PROJECT_ROOT"], "output")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        graph_name_str = "_".join(args.graph_names)
        graph_name_str = graph_name_str.replace(" ", "_")

        current_date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_file_path = os.path.join(
            log_dir,
            f"{current_date_time}_{graph_name_str}{'_inc' if args.incremental else ''}.log",
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
        graph_names_to_generate = list(GRAPH_PARAMS.keys())
    else:
        graph_names_to_generate = args.graph_names

    for graph_name in graph_names_to_generate:
        generate_graph(graph_name, args.incremental, args.preview)

    logger.info(f"log saved to {log_file_path}")
