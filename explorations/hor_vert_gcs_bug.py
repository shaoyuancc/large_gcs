import numpy as np

from large_gcs.geometry.point import Point
from large_gcs.geometry.polyhedron import Polyhedron
from large_gcs.graph.cost_constraint_factory import (
    create_2d_x_equality_edge_constraint,
    create_2d_y_equality_edge_constraint,
    create_l2norm_edge_cost,
    create_l2norm_squared_edge_cost,
)
from large_gcs.graph.graph import DefaultGraphCostsConstraints, Edge, Graph


def create_simplest_hor_vert_graph() -> Graph:
    dim = 2
    # Convex sets
    points = (
        Point((0, 0)),
        Point((4.8, 0)),
    )

    box_vert = np.array([[0, 0], [1, 0], [1, 2], [0, 2]], dtype=np.float64)
    polyhedra = (Polyhedron.from_vertices(box_vert + np.array([-0.2, 1])),)

    sets = points + polyhedra

    # Vertex names
    vertex_names = ["s", "t", "p0"]

    # Edge costs
    # There is a bug in the l2norm_squared_edge_cost!!! makes the convex restriction fail
    edge_cost = create_l2norm_squared_edge_cost(dim)
    # edge_cost = create_l2norm_edge_cost(dim)
    default_costs_constraints = DefaultGraphCostsConstraints(edge_costs=[edge_cost])
    # Add convex sets to graph
    G = Graph(default_costs_constraints)
    # G=Graph()
    G.add_vertices_from_sets(sets, names=vertex_names)
    G.set_source("s")
    G.set_target("t")

    # Edges
    edges = {
        "s": ("p0",),
        "p0": ("t",),
    }

    for u, vs in edges.items():
        for v in vs:
            print(f"Adding edge {u} -> {v}")
            G.add_edge(Edge(u, v))

    return G


def main():
    G = create_simplest_hor_vert_graph()
    result = G.solve_shortest_path()
    print("standard solve:")
    print(result.is_success)
    print(result.cost)
    print("convex restriction solve:")
    conv_restriction_result = G.solve_convex_restriction([("s", "p0"), ("p0", "t")])
    print(conv_restriction_result.is_success)
    print(conv_restriction_result.cost)


if __name__ == "__main__":
    main()
