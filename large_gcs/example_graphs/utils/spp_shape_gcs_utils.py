from typing import List, Tuple
import numpy as np
import random
from copy import deepcopy
from dataclasses import dataclass
from large_gcs.graph.graph import Graph, DefaultGraphCostsConstraints, Edge, Vertex
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from large_gcs.graph.graph import Graph
from large_gcs.geometry.point import Point
from large_gcs.geometry.polyhedron import Polyhedron
from large_gcs.geometry.ellipsoid import Ellipsoid


@dataclass
class SppShapeGcsGeneratorParams:
    dim: int = 2
    n_sets: int = 7
    source: Tuple = (-4, 0)
    target: Tuple = (4, 0)
    n_st_edges: int = 3
    workspace: List = ([-5, 5], [-5, 5])
    set_scale: float = 1.2
    k_nearest_edges: Tuple[int, int] = (0, 4)
    k_nearest_pool: int = 10
    n_polyhedron_vertices: Tuple[int, int] = (4, 20)
    random_seed: int = None
    save_path: str = "spp_shape_gcs.npy"
    should_save: bool = False


def generate_spp_shape_gcs(
    params: SppShapeGcsGeneratorParams, edge_cost_factory=None
) -> Graph:
    if params.random_seed:
        np.random.seed(params.random_seed)
    else:
        seed = np.random.randint(0, 1000000)
        print(f"Setting random seed to {seed}")
        params.random_seed = seed
        np.random.seed(seed)
    params.workspace = np.array(params.workspace)
    round_dp = 2
    # Uniformly sample n_sets within the dim dimensional workspace
    points = np.round(
        np.random.uniform(
            params.workspace[:, 0], params.workspace[:, 1], (params.n_sets, params.dim)
        ),
        round_dp,
    )

    # 2) Sort the sampled points by l2 norm distance away from the bottom left corner of the workspace
    dists = np.linalg.norm(points - params.workspace[:, 0], axis=1)
    sorted_indices = np.argsort(dists)
    points = points[sorted_indices]

    # Initialize the graph
    if edge_cost_factory:
        edge_cost = edge_cost_factory(params.dim)
        graph = Graph(DefaultGraphCostsConstraints(edge_costs=[edge_cost]))
    else:
        graph = Graph()

    polyhedra = {}
    ellipsoids = {}
    edges = {}
    # For each point, randomly choose if it's going to be a polyhedron or an ellipse
    for i, point in enumerate(points):
        if np.random.choice([True, False]):
            Q, _ = np.linalg.qr(
                np.random.randn(params.dim, params.dim)
            )  # Random orthogonal matrix
            # Diagonal matrix with entries normally distributed around 1 * set_scale
            D = np.diag(np.abs(np.random.normal(1, 0.3, params.dim))) / params.set_scale
            A = Q @ D @ Q.T
            A = np.round(A, round_dp)
            shape = Ellipsoid(center=point, A=A)
            vertex_name = f"e{i}"
            ellipsoids[vertex_name] = (point, A)
        else:
            # Create a polyhedron
            n_vertices = np.random.randint(
                params.n_polyhedron_vertices[0], params.n_polyhedron_vertices[1] + 1
            )
            vertices = np.round(
                point
                + np.random.uniform(-1, 1, (n_vertices, params.dim)) * params.set_scale,
                round_dp,
            )
            hull = ConvexHull(vertices)  # orders vertices counterclockwise
            vertices = vertices[hull.vertices]
            shape = Polyhedron(vertices=vertices)
            vertex_name = f"p{i}"
            polyhedra[vertex_name] = vertices

        # Add the vertex to the graph
        v = Vertex(shape)
        graph.add_vertex(v, vertex_name)

    # Add edges to nearby vertices
    dist_matrix = cdist(points, points)
    for i, u in enumerate(graph.vertex_names):
        nearest_indices = np.argsort(dist_matrix[i])[: params.k_nearest_pool]
        n_edges = np.random.randint(
            params.k_nearest_edges[0], params.k_nearest_edges[1]
        )
        vs = []
        for j in np.random.choice(nearest_indices, n_edges):
            v = graph.vertex_names[j]
            graph.add_edge(Edge(u, v))
            vs.append(v)
        edges[u] = vs

    # Add source and target vertices
    graph.add_vertex(Vertex(Point(params.source)), "s")
    graph.add_vertex(Vertex(Point(params.target)), "t")
    graph.set_source("s")
    graph.set_target("t")
    # Add edges from source and target to nearby vertices
    dist_matrix = cdist(np.array([params.source, params.target]), points)
    nearest_source_indices = np.argsort(dist_matrix[0])[: params.n_st_edges]
    vs = []
    for i in nearest_source_indices:
        v = graph.vertex_names[i]
        graph.add_edge(Edge("s", v))
        vs.append(v)
    edges["s"] = vs
    nearest_target_indices = np.argsort(dist_matrix[1])[: params.n_st_edges]
    for i in nearest_target_indices:
        u = graph.vertex_names[i]
        graph.add_edge(Edge(u, "t"))
        edges[u].append("t")

    if params.should_save:
        np.save(
            params.save_path,
            {
                "polyhedra": polyhedra,
                "ellipsoids": ellipsoids,
                "edges": edges,
                "params": params,
            },
        )

    return graph


def load_spp_shape_gcs(path: str, edge_cost_factory) -> Graph:
    data = np.load(path, allow_pickle=True).item()
    polyhedra = data["polyhedra"]
    ellipsoids = data["ellipsoids"]
    edges = data["edges"]
    params = data["params"]
    edge_cost = edge_cost_factory(params.dim)
    graph = Graph(DefaultGraphCostsConstraints(edge_costs=[edge_cost]))
    graph.add_vertex(Vertex(Point(params.source)), "s")
    graph.add_vertex(Vertex(Point(params.target)), "t")
    graph.set_source("s")
    graph.set_target("t")
    for vertex_name, (point, A) in ellipsoids.items():
        shape = Ellipsoid(center=point, A=A)
        graph.add_vertex(Vertex(shape), vertex_name)
    for vertex_name, vertices in polyhedra.items():
        shape = Polyhedron(vertices=vertices)
        graph.add_vertex(Vertex(shape), vertex_name)
    for u, vs in edges.items():
        for v in vs:
            graph.add_edge(Edge(u, v))

    return graph
