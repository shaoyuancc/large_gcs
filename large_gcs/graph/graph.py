from pydrake.all import (
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    Cost,
    Constraint,
    Binding,
    MathematicalProgramResult,
)
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from large_gcs.geometry.convex_set import ConvexSet


@dataclass
class ShortestPathSolution:
    cost: float
    time: float  # time to solve the problem
    path: List[
        Tuple[str, np.ndarray]
    ]  # list of vertex names and discrete coordinates in the path
    flows: List[float]  # flows along the edges (range [0, 1])
    result: MathematicalProgramResult  # result of the optimization


@dataclass
class DefaultGraphCostsConstraints:
    """
    Class to hold default costs and constraints for vertices and edges.
    """

    vertex_costs: List[Cost] = None
    vertex_constraints: List[Constraint] = None
    edge_costs: List[Cost] = None
    edge_constraints: List[Constraint] = None


@dataclass
class Vertex:
    convex_set: ConvexSet
    costs: List[Cost] = None
    constraints: List[Constraint] = None
    gcs_vertex: Optional[GraphOfConvexSets.Vertex] = None


@dataclass
class Edge:
    u: str  # source/"left" vertex of the edge
    v: str  # target/"right" vertex of the edge
    costs: List[Cost] = None
    constraints: List[Constraint] = None
    gcs_edge: Optional[GraphOfConvexSets.Edge] = None


class Graph:
    """
    Wrapper for Drake GraphOfConvexSets class.
    """

    def __init__(
        self, default_costs_constraints: DefaultGraphCostsConstraints = None
    ) -> None:
        self._default_costs_constraints = default_costs_constraints
        self.vertices = {}
        self.edges = {}
        self._source = None
        self._target = None

        self._gcs = GraphOfConvexSets()

    def add_vertex(self, vertex: Vertex, name: str = ""):
        """
        Add a vertex to the graph.
        """
        if name == "":
            name = len(self.vertices)
        assert name not in self.vertices

        vertex.gcs_vertex = self._gcs.AddVertex(vertex.convex_set.set, name)
        self.vertices[name] = vertex

        if vertex.costs is not None or vertex.constraints is not None:
            raise NotImplementedError(
                "Vertex costs and constraints not implemented yet."
            )
        elif self._default_costs_constraints is not None:
            if self._default_costs_constraints.vertex_costs is not None:
                for cost in self._default_costs_constraints.vertex_costs:
                    binding = Binding[Cost](cost, vertex.gcs_vertex.x().flatten())
                    vertex.gcs_vertex.AddCost(binding)
            if self._default_costs_constraints.vertex_constraints is not None:
                for constraint in self._default_costs_constraints.vertex_constraints:
                    binding = Binding[Constraint](
                        constraint, vertex.gcs_vertex.x().flatten()
                    )
                    vertex.gcs_vertex.AddConstraint(binding)

    def add_vertices_from_sets(
        self, sets: List[ConvexSet], costs=None, constraints=None, names=None
    ):
        """
        Add vertices to the graph. Each vertex is a convex set.
        """
        if names is None:
            names = [None] * len(sets)
        else:
            assert len(sets) == len(names)
        if costs is None:
            costs = [None] * len(sets)
        else:
            assert len(costs) == len(sets)
        if constraints is None:
            constraints = [None] * len(sets)
        else:
            assert len(constraints) == len(sets)

        for set, name, cost_list, constraint_list in zip(
            sets, names, costs, constraints
        ):
            self.add_vertex(Vertex(set, cost_list, constraint_list), name)

    def add_edge(self, edge: Edge):
        """
        Add an edge to the graph.
        """

        edge.gcs_edge = self._gcs.AddEdge(
            self.vertices[edge.u].gcs_vertex, self.vertices[edge.v].gcs_vertex
        )
        self.edges[(edge.u, edge.v)] = edge

        if edge.costs is not None or edge.constraints is not None:
            raise NotImplementedError("Edge costs and constraints not implemented yet.")
        elif self._default_costs_constraints is not None:
            if self._default_costs_constraints.edge_costs is not None:
                for cost in self._default_costs_constraints.edge_costs:
                    x = np.array([edge.gcs_edge.xu(), edge.gcs_edge.xv()]).flatten()
                    binding = Binding[Cost](cost, x)
                    edge.gcs_edge.AddCost(binding)
            if self._default_costs_constraints.edge_constraints is not None:
                for constraint in self._default_costs_constraints.edge_constraints:
                    x = np.array([edge.gcs_edge.xu(), edge.gcs_edge.xv()]).flatten()
                    binding = Binding[Constraint](constraint, x)
                    edge.gcs_edge.AddConstraint(binding, x)

    def add_edges_from_vertex_names(
        self,
        us: List[str],
        vs: List[str],
        costs: List[List[Cost]] = None,
        constraints: List[List[Constraint]] = None,
    ):
        """
        Add edges to the graph.
        """

        for u, v, cost_list, constraint_list in zip(us, vs, costs, constraints):
            self.add_edge(Edge(u, v, cost_list, constraint_list))

    def set_source(self, vertex_name: str):
        assert vertex_name in self.vertices
        self._source = vertex_name

    def set_target(self, vertex_name: str):
        assert vertex_name in self.vertices
        self._target = vertex_name

    def solve_shortest_path(self, use_convex_relaxation=False) -> ShortestPathSolution:
        """
        Solve the shortest path problem.
        """
        assert self._source is not None
        assert self._target is not None

        options = GraphOfConvexSetsOptions()

        options.convex_relaxation = use_convex_relaxation
        if use_convex_relaxation is True:
            options.preprocessing = True
            options.max_rounded_paths = 10

        print("Solving GCS problem...")
        result = self._gcs.SolveShortestPath(
            self.vertices[self._source].gcs_vertex,
            self.vertices[self._target].gcs_vertex,
            options,
        )
        assert result.is_success()
        print("Result is success!")

        return self._parse_shortest_path_result(result)

    def _parse_shortest_path_result(
        self, result: MathematicalProgramResult
    ) -> ShortestPathSolution:
        cost = result.get_optimal_cost()
        time = result.get_solver_details().optimizer_time

        flow_variables = [e.phi() for e in self._gcs.Edges()]
        flows = [result.GetSolution(p) for p in flow_variables]
        edge_path = []
        for k, flow in enumerate(flows):
            if flow >= 0.99:
                edge_path.append(self.edge_keys[k])
        # Edges are in order they were added to the graph and not in order of the path
        vertex_path = self._convert_active_edges_to_vertex_path(
            self.source_name, edge_path
        )
        path = [
            (v, result.GetSolution(self.vertices[v].gcs_vertex.x()))
            for v in vertex_path
        ]

        return ShortestPathSolution(cost, time, path, flows, result)

    @staticmethod
    def _convert_active_edges_to_vertex_path(source_name, edges):
        # Create a dictionary where the keys are the vertices and the values are their neighbors
        neighbors = {u: v for u, v in edges}
        # Start with the source vertex
        path = [source_name]

        # While the last vertex in the path has a neighbor
        while path[-1] in neighbors:
            # Add the neighbor to the path
            path.append(neighbors[path[-1]])

        assert len(path) == len(edges) + 1, "Path length does not match number of edges"

        return path

    def plot_sets(self, **kwargs):
        plt.rc("axes", axisbelow=True)
        plt.gca().set_aspect("equal")
        for v in self.vertices.values():
            v.convex_set.plot(**kwargs)

    def plot_edges(self, **kwargs):
        options = {
            "color": "k",
            "zorder": 2,
            "arrowstyle": "->, head_width=3, head_length=8",
        }
        options.update(kwargs)
        for edge in self.edge_keys:
            tail = self.vertices[edge[0]].convex_set.center
            head = self.vertices[edge[1]].convex_set.center
            arrow = patches.FancyArrowPatch(tail, head, **options)
            plt.gca().add_patch(arrow)

    def plot_set_labels(self, labels=None, **kwargs):
        options = {"c": "b"}
        options.update(kwargs)
        if labels is None:
            labels = self.vertex_names

        for v, label in zip(self.vertices.values(), labels):
            plt.text(*v.convex_set.center, label, **options)

    def plot_edge_labels(self, labels, **kwargs):
        options = {"c": "r", "va": "top"}
        options.update(kwargs)
        for edge, label in zip(self.edges, labels):
            center = (
                self.vertices[edge[0]].convex_set.center
                + self.vertices[edge[1]].convex_set.center
            ) / 2
            plt.text(*center, label, **options)

    def plot_points(self, x, **kwargs):
        options = {"marker": "o", "facecolor": "w", "edgecolor": "k", "zorder": 3}
        options.update(kwargs)
        plt.scatter(*x.T, **options)

    def plot_path(self, path: List[Tuple[str, np.ndarray]], **kwargs):
        options = {
            "color": "g",
            "marker": "o",
            "markeredgecolor": "k",
            "markerfacecolor": "w",
        }
        options.update(kwargs)
        plt.plot(*np.array([x for _, x in path]).T, **options)

    def edge_key_index(self, edge_key):
        return self.edge_keys.index(edge_key)

    def edge_indices(self, edge_keys):
        return [self.edge_keys.index(key) for key in edge_keys]

    def vertex_name_index(self, vertex_name):
        return self.vertex_names.index(vertex_name)

    def vertex_name_indices(self, vertex_names):
        return [self.vertex_names.index(name) for name in vertex_names]

    @property
    def vertex_names(self):
        return list(self.vertices.keys())

    @property
    def edge_keys(self):
        return list(self.edges.keys())

    @property
    def source_name(self):
        return (
            self.vertex_names[self.vertices[0]]
            if self._source is None
            else self._source
        )

    @property
    def target_name(self):
        return (
            self.vertex_names[self.vertices[-1]]
            if self._target is None
            else self._target
        )

    @property
    def source(self):
        return self.vertices[self.source_name]

    @property
    def target(self):
        return self.vertices[self.target_name]

    @property
    def dimension(self):
        assert len(set(V.convex_set.dimension for V in self.vertices.values())) == 1
        return self.vertices[0].convex_set.dimension

    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def n_edges(self):
        return len(self.edges)
