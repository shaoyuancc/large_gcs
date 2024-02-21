import numpy as np

from large_gcs.algorithms.gcs_astar_reachability import GcsAstarReachability
from large_gcs.cost_estimators.shortcut_edge_ce import ShortcutEdgeCE
from large_gcs.graph.contact_cost_constraint_factory import (
    contact_shortcut_edge_cost_factory_over_obj_weighted,
)
from large_gcs.graph.cost_constraint_factory import shortcut_edge_cost_factory
from large_gcs.graph_generators.hor_vert_gcs import (
    create_polyhedral_hor_vert_graph,
    create_simplest_hor_vert_graph,
)

tol = 1e-3


def test_gcs_astar_reachability_polyhedra_hor_vert():
    g = create_polyhedral_hor_vert_graph()
    cost_estimator_se = ShortcutEdgeCE(g, shortcut_edge_cost_factory)
    alg = GcsAstarReachability(g, cost_estimator_se, num_samples_per_vertex=3)
    sol = alg.run()
    ambient_path = np.array([[0.5, 0.5], [0.5, 3.9], [4.5, 3.9], [4.5, 0.5]])
    vertex_path = ["s", "p1", "p2", "t"]
    assert np.allclose(sol.ambient_path, ambient_path)
    assert sol.vertex_path == vertex_path
    assert np.isclose(sol.cost, 10.8, atol=tol)
