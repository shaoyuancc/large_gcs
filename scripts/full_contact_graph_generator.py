import argparse
import logging
import os
import time

import numpy as np

from large_gcs.contact.contact_regions_set import ContactRegionParams
from large_gcs.graph_generators.contact_graph_generator import (
    ContactGraphGenerator,
    ContactGraphGeneratorParams,
)

logger = logging.getLogger(__name__)


def main(graph_name: str):
    params = get_graph_params(graph_name)

    # Create log directory relative to the script location
    log_dir = os.path.join(os.environ["PROJECT_ROOT"], "output")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    current_date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(log_dir, f"{current_date_time}_{params.name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    )
    logging.getLogger("large_gcs").setLevel(logging.DEBUG)
    logging.getLogger("drake").setLevel(logging.WARNING)
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    logger.info(f"log saved to {log_file_path}")

    start_time = time.time()
    logger.info(f"Starting graph creation at: {start_time}")

    generator = ContactGraphGenerator(params)
    graph = generator.generate()
    end_time = time.time()
    logger.info(f"Graph creation took: {end_time - start_time} seconds")
    logger.info(
        f"duration in H:M:S {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}"
    )
    logger.info(graph.params)
    logger.info(f"log saved to {log_file_path}")


def get_graph_params(graph_name: str) -> ContactGraphGeneratorParams:
    if graph_name == "cg_simple_3":
        ws = 3
        target_regions = [
            ContactRegionParams(
                region_vertices=[[1.5, -1], [1.5, 1], [3, 1], [3, -1]], obj_indices=[0]
            ),
        ]
        params = ContactGraphGeneratorParams(
            name="cg_simple_3",
            obs_vertices=[],
            obj_vertices=[
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) + np.array([2.5, 0.5])
            ],
            rob_vertices=[np.array([[-1, -1], [-1.5, -0.5], [-1.2, -1.5]])],
            source_obj_pos=[[0, 0]],
            source_rob_pos=[[-2, -2]],
            target_region_params=target_regions,
            n_pos_per_set=2,
            workspace=[[-ws, ws], [-ws, ws]],
            should_add_const_edge_cost=True,
            should_use_l1_norm_vertex_cost=True,
        )
    elif graph_name == "cg_maze_b1":
        params = ContactGraphGeneratorParams(
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
            target_obj_pos=None,
            target_rob_pos=None,
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
        )
    elif graph_name == "cg_trichal4":
        ws_x = 5
        ws_y = 4
        target_region_params = [
            ContactRegionParams(
                region_vertices=[[-3, -0.75], [-3, 1], [-1, -0.75], [-1, 1]],
                obj_indices=[0],
            ),
        ]
        params = ContactGraphGeneratorParams(
            name="cg_trichal4",
            obs_vertices=[[[-1, 2], [-1, -1], [2, 2]]],
            obj_vertices=[[[1, 0.5], [1, -0.5], [2, -0.5], [2, 0.5]]],
            rob_vertices=[[[3, 1], [3, 0], [3.5, 0]]],
            source_obj_pos=[[3.25, 0]],
            source_rob_pos=[[1.5, 0.5]],
            target_region_params=target_region_params,
            n_pos_per_set=2,
            workspace=[[-ws_x, ws_x], [-ws_y, ws_y]],
            should_add_const_edge_cost=True,
            should_use_l1_norm_vertex_cost=True,
        )
    elif graph_name == "cg_stackpush_d2":
        ws_x = 7
        ws_y = 5
        target_region_params = [
            ContactRegionParams(
                region_vertices=[[3, 0], [3, -3], [6, 0], [6, -3]],
                obj_indices=[0, 1, 2],
            ),
        ]
        params = ContactGraphGeneratorParams(
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
            target_region_params=target_region_params,
            n_pos_per_set=2,
            workspace=[[-ws_x, ws_x], [-ws_y, ws_y]],
        )
    else:
        raise ValueError(f"Unknown graph name: {graph_name}")
    return params


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()

    parser = argparse.ArgumentParser(description="Generate contact graph")
    parser.add_argument("graph_name", type=str, help="Name of the graph to generate")
    args = parser.parse_args()

    main(args.graph_name)
