import numpy as np
import pytest

from large_gcs.algorithms.deprecated.gcs_dijkstra import GcsDijkstra
from large_gcs.graph.cost_constraint_factory import create_l2norm_edge_cost
from large_gcs.graph_generators.spp_gcs import create_spp_2d_graph

pytestmark = pytest.mark.skip(reason="GcsDijkstra is deprecated, skipping tests.")


def test_dijkstra_spp_2d():
    G = create_spp_2d_graph(create_l2norm_edge_cost)
    sol = GcsDijkstra(G).run(animate=False)
    ambient_path = np.array(
        [
            [0.0, 0.0],
            [2.18127433, 0.34299636],
            [6.53338475, 1.02759693],
            [8.99999998, 0.0],
        ]
    )
    vertex_path = np.array(["s", "p0", "e1", "t"])
    assert np.isclose(sol.cost, 9.285808987971189)
    assert np.allclose(sol.ambient_path, ambient_path, atol=1e-2)
    assert np.array_equal(sol.vertex_path, vertex_path)
