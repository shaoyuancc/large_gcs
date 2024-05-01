import numpy as np

from large_gcs.geometry.point import Point
from large_gcs.geometry.polyhedron import Polyhedron
from large_gcs.graph.cost_constraint_factory import (
    create_2d_x_equality_edge_constraint,
    create_2d_y_equality_edge_constraint,
    create_l1norm_edge_cost,
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
    polyhedra = (
        Polyhedron.from_vertices(box_vert + np.array([-0.2, 1])),
        Polyhedron.from_vertices(box_vert + np.array([-0.5, 3.5])),
        Polyhedron.from_vertices([[2, 0.5], [2.5, -0.5], [5, 5], [4.5, 6]]),
    )

    sets = points + polyhedra

    # Vertex names
    vertex_names = ["s", "t"]
    vertex_names += [f"p{i}" for i in range(len(polyhedra))]

    # Edge costs
    # edge_cost = create_l2norm_squared_edge_cost(dim)
    edge_cost = create_l2norm_edge_cost(dim)
    default_costs_constraints = DefaultGraphCostsConstraints(edge_costs=[edge_cost])
    # Add convex sets to graph
    G = Graph(default_costs_constraints)
    G.add_vertices_from_sets(sets, names=vertex_names)
    G.set_source("s")
    G.set_target("t")

    # Edges
    vert_edges = {
        "s": ("p0", "p1"),
        "p2": ("t",),
    }
    hor_edges = {
        "p0": ("p2",),
        "p1": ("p2",),
    }

    def add_edges(edges, constraints):
        for u, vs in edges.items():
            for v in vs:
                # print(f"Adding edge {u} -> {v}")
                G.add_edge(Edge(u, v, constraints=constraints))

    vert_constraint = [create_2d_x_equality_edge_constraint()]
    hor_constraint = [create_2d_y_equality_edge_constraint()]
    add_edges(vert_edges, vert_constraint)
    add_edges(hor_edges, hor_constraint)
    return G


def create_polyhedral_hor_vert_graph() -> Graph:
    dim = 2
    # Convex sets
    box_vert = np.array([[0, 0], [1, 0], [1, 2], [0, 2]], dtype=np.float64)
    sets = (
        # source
        Polyhedron.from_vertices([[-0.5, 0], [1, 0], [1, 0.5], [-0.5, 0.5]]),
        # target
        Polyhedron.from_vertices([[4.5, 0], [5.5, 0], [5.5, 0.5], [4.5, 0.5]]),
        # intermediate sets
        Polyhedron.from_vertices(box_vert + np.array([-0.2, 1])),
        Polyhedron.from_vertices(box_vert + np.array([-0.5, 3.5])),
        Polyhedron.from_vertices([[2, 0.5], [2.5, -0.5], [5, 5], [4.5, 6]]),
    )

    # Vertex names
    vertex_names = ["s", "t"]
    vertex_names += [f"p{i}" for i in range(len(sets) - 2)]

    # Edge costs
    # edge_cost = create_l2norm_squared_edge_cost(dim)
    edge_cost = create_l2norm_edge_cost(dim)
    default_costs_constraints = DefaultGraphCostsConstraints(edge_costs=[edge_cost])
    # Add convex sets to graph
    G = Graph(default_costs_constraints)
    G.add_vertices_from_sets(sets, names=vertex_names)
    G.set_source("s")
    G.set_target("t")

    # Edges
    vert_edges = {
        "s": ("p0", "p1"),
        "p2": ("t",),
    }
    hor_edges = {
        "p0": ("p2",),
        "p1": ("p2",),
    }

    def add_edges(edges, constraints):
        for u, vs in edges.items():
            for v in vs:
                # print(f"Adding edge {u} -> {v}")
                G.add_edge(Edge(u, v, constraints=constraints))

    vert_constraint = [create_2d_x_equality_edge_constraint()]
    hor_constraint = [create_2d_y_equality_edge_constraint()]
    add_edges(vert_edges, vert_constraint)
    add_edges(hor_edges, hor_constraint)
    return G


def create_intermediate_line_hor_vert_graph() -> Graph:
    dim = 2
    # Convex sets
    points = (
        Point((0, 0)),
        Point((4.8, 0)),
    )

    box_vert = np.array([[0, 0], [1, 0], [1, 2], [0, 2]], dtype=np.float64)
    polyhedra = (
        Polyhedron.from_vertices(box_vert + np.array([-0.2, 1])),
        Polyhedron.from_vertices(box_vert + np.array([-0.5, 3.5])),
        Polyhedron.from_vertices([[0, 5.6], [2, 6]]),
        Polyhedron.from_vertices([[2, 0.5], [2.5, -0.5], [6, 6], [4, 6]]),
    )

    sets = points + polyhedra

    # Vertex names
    vertex_names = ["s", "t"]
    vertex_names += [f"p{i}" for i in range(len(polyhedra))]

    # Edge costs
    # edge_cost = create_l2norm_squared_edge_cost(dim)
    edge_cost = create_l2norm_edge_cost(dim)
    default_costs_constraints = DefaultGraphCostsConstraints(edge_costs=[edge_cost])
    # Add convex sets to graph
    G = Graph(default_costs_constraints)
    G.add_vertices_from_sets(sets, names=vertex_names)
    G.set_source("s")
    G.set_target("t")

    # Edges
    vert_edges = {
        "s": ("p0", "p1"),
        "p0": ("p2",),
        "p1": ("p2",),
        "p3": ("t",),
    }
    hor_edges = {
        "p2": ("p3",),
    }

    def add_edges(edges, constraints):
        for u, vs in edges.items():
            for v in vs:
                # print(f"Adding edge {u} -> {v}")
                G.add_edge(Edge(u, v, constraints=constraints))

    vert_constraint = [create_2d_x_equality_edge_constraint()]
    hor_constraint = [create_2d_y_equality_edge_constraint()]
    add_edges(vert_edges, vert_constraint)
    add_edges(hor_edges, hor_constraint)
    return G

def create_polyhedral_hor_vert_b_graph() -> Graph:
    dim = 2
    # Convex sets
    box_vert = np.array([[0, 0], [1, 0], [1, 1.5], [0, 1.5]], dtype=np.float64)
    sets = (
        # source
        Polyhedron.from_vertices([[-0.5, 0], [1, 0], [1, 0.5], [-0.5, 0.5]]),
        # target
        Polyhedron.from_vertices([[4.5, 0], [5.5, 0], [5.5, 0.5], [4.5, 0.5]]),
        # intermediate sets
        Polyhedron.from_vertices(box_vert + np.array([-0.7, 3])),
        Polyhedron.from_vertices(box_vert + np.array([-0.5, 7.5])),
        # Diagonal set
        Polyhedron.from_vertices([[2, 3], [2.5, 3], [5, 9], [4.5, 9]]),
        # Cluster of 3
        Polyhedron.from_vertices([[1.5, 0.3], [2.5, 0.3], [1.5, 0.7]]),
        Polyhedron.from_vertices([[2.3, 0], [3.3, 0], [2.3, 0.2]]),
        Polyhedron.from_vertices([[3, 0.3], [4, 0.3], [3, 0.7]]),
        # Bottom row
        Polyhedron.from_vertices([[0.5, -1], [1, -1], [1, -0.5], [0.5, -0.5]]),
        Polyhedron.from_vertices([[3,-1], [6,-1], [6,-0.5], [3,-0.5]]),
    )

    # Vertex names
    vertex_names = ["s", "t"]
    vertex_names += [f"p{i}" for i in range(len(sets) - 2)]

    # Edge costs
    # edge_cost = create_l2norm_squared_edge_cost(dim)
    edge_cost = create_l1norm_edge_cost(dim)
    default_costs_constraints = DefaultGraphCostsConstraints(edge_costs=[edge_cost])
    # Add convex sets to graph
    G = Graph(default_costs_constraints)
    G.add_vertices_from_sets(sets, names=vertex_names)
    G.set_source("s")
    G.set_target("t")

    # Edges
    vert_edges = {
        "s": ("p0", "p1", "p6"),
        "p2": ("t",),
        "p3": ("p2",),
        "p4": ("p2",),
        "p5": ("p2",),
        "p7": ("p2",),
    }
    hor_edges = {
        "p0": ("p2",),
        "p1": ("p2",),
        "p6": ("p7",),
    }

    def add_edges(edges, constraints):
        for u, vs in edges.items():
            for v in vs:
                # print(f"Adding edge {u} -> {v}")
                G.add_edge(Edge(u, v, constraints=constraints))

    vert_constraint = [create_2d_x_equality_edge_constraint()]
    hor_constraint = [create_2d_y_equality_edge_constraint()]
    add_edges(vert_edges, vert_constraint)
    add_edges(hor_edges, hor_constraint)
    return G