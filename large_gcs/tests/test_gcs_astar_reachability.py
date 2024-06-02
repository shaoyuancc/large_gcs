import numpy as np

from large_gcs.algorithms.gcs_astar_reachability import GcsAstarReachability
from large_gcs.cost_estimators.shortcut_edge_ce import ShortcutEdgeCE
from large_gcs.domination_checkers.reaches_new_sampling import ReachesNewSampling
from large_gcs.graph.cost_constraint_factory import shortcut_edge_cost_factory
from large_gcs.graph_generators.hor_vert_gcs import create_polyhedral_hor_vert_graph

tol = 1e-3


def test_gcs_astar_reachability_polyhedra_hor_vert():
    g = create_polyhedral_hor_vert_graph()
    cost_estimator_se = ShortcutEdgeCE(g, shortcut_edge_cost_factory)
    domination_checker = ReachesNewSampling(graph=g, num_samples_per_vertex=5)
    alg = GcsAstarReachability(
        g, cost_estimator_se, domination_checker=domination_checker
    )
    sol = alg.run()
    ambient_path = np.array([[0.5, 0.5], [0.5, 3.9], [4.5, 3.9], [4.5, 0.5]])
    vertex_path = ["s", "p1", "p2", "t"]
    assert np.allclose(sol.ambient_path, ambient_path)
    assert sol.vertex_path == vertex_path
    assert np.isclose(sol.cost, 10.8, atol=tol)
