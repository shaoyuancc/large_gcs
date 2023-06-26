from typing import List, Tuple
import numpy as np
from copy import deepcopy
from tqdm import tqdm
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
    # Dimension of the ambient space of the graph
    dim: int = 2
    # Number of convex sets
    n_sets: int = 7
    # Source and target points
    source: Tuple = (-4, 0)
    target: Tuple = (4, 0)
    # Number of edges from the source and to the target
    n_st_edges: int = 3
    # Workspace bounds from which samples are drawn
    samples_workspace: List = ([-5, 5], [-5, 5])
    # Scaling factor for each of the convex sets
    set_scale: float = 1.2
    # Lower and upper bounds on the number of edges from each convex set
    k_nearest_edges: Tuple[int, int] = (0, 4)
    # Size of the pool of nearest convex sets to choose from to form edges
    k_nearest_pool: int = 10
    # Lower and upper bounds on the number of vertices of each polyhedra
    n_polyhedron_vertices: Tuple[int, int] = (4, 20)
    # To be able to reproduce the same graph
    random_seed: int = None
    # Path to save the graph, should be a .npy file
    save_path: str = "spp_shape_gcs.npy"
    # Whether to save the graph as a .npy file
    should_save: bool = False

    def __post_init__(self):
        self.source = np.array(self.source)
        self.target = np.array(self.target)
        self.samples_workspace = np.array(self.samples_workspace)
        self.k_nearest_edges = np.array(self.k_nearest_edges)
        self.n_polyhedron_vertices = np.array(self.n_polyhedron_vertices)

        assert self.source.shape == self.target.shape == (self.dim,)
        assert self.samples_workspace.shape == (self.dim, 2)
        assert self.k_nearest_edges.shape == (2,)
        assert self.n_polyhedron_vertices.shape == (2,)


def generate_spp_shape_gcs(
    params: SppShapeGcsGeneratorParams, edge_cost_factory=None
) -> Graph:
    if params.random_seed:
        np.random.seed(params.random_seed)
    else:
        seed = np.random.randint(0, 1000000)
        params.random_seed = seed
        np.random.seed(seed)
    params.samples_workspace = np.array(params.samples_workspace)
    round_dp = 2

    # Uniformly sample n_sets within the dim dimensional workspace
    print("Sampling points uniformly within the workspace...")
    samples = np.round(
        np.random.uniform(
            params.samples_workspace[:, 0],
            params.samples_workspace[:, 1],
            (params.n_sets, params.dim),
        ),
        round_dp,
    )

    # Sort the sampled points by l2 norm distance away from the bottom left corner of the workspace
    dists = np.linalg.norm(samples - params.samples_workspace[:, 0], axis=1)
    sorted_indices = np.argsort(dists)
    samples = samples[sorted_indices]
    vertex_names_by_samples = []

    # Initialize the graph
    workspace_bounds = params.samples_workspace * params.set_scale
    if edge_cost_factory:
        edge_cost = edge_cost_factory(params.dim)
        graph = Graph(
            DefaultGraphCostsConstraints(edge_costs=[edge_cost]),
            workspace_bounds=workspace_bounds,
        )
    else:
        graph = Graph(workspace=workspace_bounds)

    points = {}
    polyhedra = {}
    ellipsoids = {}
    edges = {}

    # Add source and target vertices
    points["s"] = params.source
    points["t"] = params.target

    # For each point, randomly choose if it's going to be a polyhedron or an ellipse
    print("Processing samples into convex sets...")
    for i in tqdm(range(len(samples))):
        sample = samples[i]
        if np.random.choice([True, False]):
            Q, _ = np.linalg.qr(
                np.random.randn(params.dim, params.dim)
            )  # Random orthogonal matrix
            # Diagonal matrix with entries normally distributed around 1 * set_scale
            D = np.diag(np.abs(np.random.normal(1, 0.3, params.dim))) / params.set_scale
            A = Q @ D @ Q.T
            A = np.round(A, round_dp)
            vertex_name = f"e{i}"
            ellipsoids[vertex_name] = (sample, A)
        else:
            # Create a polyhedron
            n_vertices = np.random.randint(
                params.n_polyhedron_vertices[0], params.n_polyhedron_vertices[1] + 1
            )
            vertices = np.round(
                sample
                + np.random.uniform(-1, 1, (n_vertices, params.dim)) * params.set_scale,
                round_dp,
            )
            hull = ConvexHull(vertices)  # orders vertices counterclockwise
            vertices = vertices[hull.vertices]
            shape = Polyhedron(vertices=vertices)
            vertex_name = f"p{i}"
            polyhedra[vertex_name] = vertices
        vertex_names_by_samples.append(vertex_name)
    vertex_names_by_samples = np.array(vertex_names_by_samples)

    # Add edges from source and target to nearby vertices
    print("Calculating distance matrix...")
    st_dist_matrix = cdist(np.array([params.source, params.target]), samples)
    nearest_source_indices = np.argsort(st_dist_matrix[0])[: params.n_st_edges]
    edges["s"] = vertex_names_by_samples[nearest_source_indices]

    # Add edges to nearby vertices
    dist_matrix = cdist(samples, samples)
    print("Adding edges to vertices...")
    for i in tqdm(range(len(vertex_names_by_samples))):
        u = vertex_names_by_samples[i]
        nearest_indices = np.argsort(dist_matrix[i])[1 : params.k_nearest_pool + 1]
        n_edges = np.random.randint(
            params.k_nearest_edges[0], params.k_nearest_edges[1]
        )
        edges[u] = vertex_names_by_samples[
            np.random.choice(nearest_indices, n_edges, replace=False)
        ]

    nearest_target_indices = np.argsort(st_dist_matrix[1])[: params.n_st_edges]
    for i in tqdm(range(len(nearest_target_indices))):
        u = vertex_names_by_samples[nearest_target_indices[i]]
        if u in edges:
            edges[u] = np.append(edges[u], "t")
        else:
            edges[u] = np.array(["t"])

    if params.should_save:
        print("Saving graph data...")
        np.save(
            params.save_path,
            {
                "points": points,
                "polyhedra": polyhedra,
                "ellipsoids": ellipsoids,
                "edges": edges,
                "params": params,
            },
        )

    print("Adding vertices and edges to graph...")
    graph = _add_vertices_edges_to_graph(points, ellipsoids, polyhedra, edges, graph)
    return graph


def load_spp_shape_gcs(path: str, edge_cost_factory) -> Graph:
    data = np.load(path, allow_pickle=True).item()
    points = data["points"]
    polyhedra = data["polyhedra"]
    ellipsoids = data["ellipsoids"]
    edges = data["edges"]
    params = data["params"]
    edge_cost = edge_cost_factory(params.dim)
    workspace_bounds = params.samples_workspace * params.set_scale
    graph = Graph(
        DefaultGraphCostsConstraints(edge_costs=[edge_cost]),
        workspace_bounds=workspace_bounds,
    )
    graph = _add_vertices_edges_to_graph(points, ellipsoids, polyhedra, edges, graph)
    return graph


from tqdm import tqdm


def _add_vertices_edges_to_graph(points, ellipsoids, polyhedra, edges, graph):
    print("Adding points as vertices to the graph...")
    for vertex_name in tqdm(points.keys()):
        point = points[vertex_name]
        shape = Point(point)
        graph.add_vertex(Vertex(shape), vertex_name)

    print("Adding ellipsoids as vertices to the graph...")
    for vertex_name in tqdm(ellipsoids.keys()):
        center, A = ellipsoids[vertex_name]
        shape = Ellipsoid(center=center, A=A)
        graph.add_vertex(Vertex(shape), vertex_name)

    print("Adding polyhedra as vertices to the graph...")
    for vertex_name in tqdm(polyhedra.keys()):
        vertices = polyhedra[vertex_name]
        shape = Polyhedron(vertices=vertices)
        graph.add_vertex(Vertex(shape), vertex_name)

    graph.set_source("s")
    graph.set_target("t")

    print("Adding edges to the graph...")
    for u in tqdm(edges.keys()):
        vs = edges[u]
        for v in vs:
            graph.add_edge(Edge(u, v))

    return graph
