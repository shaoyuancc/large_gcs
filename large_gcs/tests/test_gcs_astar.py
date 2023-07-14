from large_gcs.example_graphs.spp_gcs import create_spp_2d_graph
from large_gcs.graph.cost_factory import (
    create_l2norm_edge_cost,
    create_l2norm_squared_edge_cost,
)
from large_gcs.algorithms.gcs_astar import GcsAstar
import numpy as np

tol = 1e-3


def test_astar_spp_2d():
    G = create_spp_2d_graph(create_l2norm_edge_cost)
    sol = GcsAstar(G).run(animate=False)
    ambient_path = np.array(
        [[0.0, 0.0], [2.25006365, 0.35391994], [6.5333202, 1.0276048], [9.0, 0.0]]
    )
    vertex_path = np.array(["s", "p0", "e1", "t"])
    assert np.isclose(sol.cost, 9.285808987971189, atol=tol)
    assert np.allclose(sol.ambient_path, ambient_path, atol=tol)
    assert np.array_equal(sol.vertex_path, vertex_path)
