import numpy as np
import pytest

from large_gcs.algorithms.gcs_astar_convex_restriction import GcsAstarConvexRestriction
from large_gcs.algorithms.search_algorithm import ReexploreLevel
from large_gcs.cost_estimators.factored_collision_free_ce import FactoredCollisionFreeCE
from large_gcs.graph.contact_cost_constraint_factory import (
    contact_shortcut_edge_cost_factory_over_obj_weighted,
)
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.graph import ShortestPathSolution
from large_gcs.graph.incremental_contact_graph import IncrementalContactGraph
from large_gcs.graph_generators.contact_graph_generator import (
    ContactGraphGeneratorParams,
)

tol = 1e-3


def test_cfree_graph_all_feasible_trichal2_inc():
    graph_file = ContactGraphGeneratorParams.inc_graph_file_path_from_name(
        "cg_trichal2"
    )
    cg = IncrementalContactGraph.load_from_file(
        graph_file,
        should_incl_simul_mode_switches=False,
        should_add_const_edge_cost=True,
        should_add_gcs=True,
    )
    cost_estimator = FactoredCollisionFreeCE(
        cg,
        use_combined_gcs=False,
        add_transition_cost=True,
        obj_multiplier=100.0,
    )
    for cfree_graph_index, cfree_graph in cost_estimator._cfree_graphs.items():
        for v in cfree_graph.vertex_names:
            if v == cfree_graph.target_name:
                continue
            cfree_graph.set_source(v)
            sol = cfree_graph.solve_shortest_path()
            assert (
                sol.is_success
            ), f"Failed to find path from {v} to {cfree_graph.target_name} in cfree graph {cfree_graph_index}"
