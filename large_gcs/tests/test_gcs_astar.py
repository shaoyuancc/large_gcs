import numpy as np
import pytest

from large_gcs.algorithms.deprecated.gcs_astar import GcsAstar
from large_gcs.geometry.point import Point
from large_gcs.graph.cost_constraint_factory import create_l2norm_edge_cost
from large_gcs.graph.graph import DefaultGraphCostsConstraints, Edge, Graph
from large_gcs.graph_generators.spp_gcs import create_spp_2d_graph

tol = 1e-3

pytestmark = pytest.mark.skip(reason="GcsAstar is deprecated, skipping tests.")


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


def test_astar_early_terminate_node_with_edge_to_target_explored():
    dim = 2
    sets = (
        Point((0, 0)),
        Point((5, 5)),
        Point((5, 0)),
        Point((7, 0)),
        Point((10, 0)),
    )
    vertex_names = ["s", "a", "b", "c", "t"]
    edge_cost = create_l2norm_edge_cost(dim)
    default_costs_constraints = DefaultGraphCostsConstraints(edge_costs=[edge_cost])
    G = Graph(default_costs_constraints)
    G.add_vertices_from_sets(sets, names=vertex_names)
    G.set_source("s")
    G.set_target("t")
    edges = {
        "s": ("a", "b"),
        "a": ("t",),
        "b": ("c",),
        "c": ("t",),
    }
    for u, vs in edges.items():
        for v in vs:
            G.add_edge(Edge(u, v))
    sol = GcsAstar(G).run()
    vertex_path = np.array(["s", "b", "c", "t"])
    assert np.array_equal(sol.vertex_path, vertex_path)
