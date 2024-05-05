from large_gcs.geometry.polyhedron import Polyhedron
from large_gcs.graph.cost_constraint_factory import create_l1norm_edge_cost
from large_gcs.graph.graph import DefaultGraphCostsConstraints, Edge, Graph


def create_simple_1d_graph() -> Graph:
    dim = 1
    # Convex sets
    sets = (
        # s
        Polyhedron.from_vertices([[0], [1]]),
        # t
        Polyhedron.from_vertices([[2], [3]]),
        # p0
        Polyhedron.from_vertices([[0.5], [1.5]]),
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
    hor_edges = {
        "s": ("t",),
        "p0": ("t",),
    }

    def add_edges(edges, constraints):
        for u, vs in edges.items():
            for v in vs:
                # print(f"Adding edge {u} -> {v}")
                G.add_edge(Edge(u, v, constraints=constraints))

    add_edges(hor_edges, None)
    return G
