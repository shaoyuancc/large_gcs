import logging

import numpy as np
from IPython.display import HTML
from matplotlib import pyplot as plt

from large_gcs.algorithms.gcs_astar_convex_restriction import GcsAstarConvexRestriction
from large_gcs.algorithms.search_algorithm import ReexploreLevel
from large_gcs.contact.contact_regions_set import ContactRegionParams
from large_gcs.cost_estimators.shortcut_edge_ce import ShortcutEdgeCE
from large_gcs.graph.contact_cost_constraint_factory import (
    contact_shortcut_edge_cost_factory_over_obj_weighted,
    contact_shortcut_edge_cost_factory_under,
)
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.incremental_contact_graph import IncrementalContactGraph
from large_gcs.graph_generators.contact_graph_generator import (
    ContactGraphGenerator,
    ContactGraphGeneratorParams,
)

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    # Rest of your code

    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("large_gcs").setLevel(logging.INFO)
    logging.getLogger("drake").setLevel(logging.WARNING)

    ws_x = 7
    ws_y = 5
    target_region_params = [
        ContactRegionParams(
            region_vertices=[[3, 0], [3, -3], [6, 0], [6, -3]], obj_indices=[0, 1, 2]
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

    generator = ContactGraphGenerator(params)
    graph = generator.generate()
    print(graph.params)
