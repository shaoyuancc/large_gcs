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
from copy import copy
from graphviz import Digraph
from large_gcs.geometry.convex_set import ConvexSet


@dataclass
class ShortestPathSolution:
    cost: float
    # Time to solve the optimization problem
    time: float
    # List of vertex names and discrete coordinates in the path
    vertex_path: List[str]
    ambient_path: List[np.ndarray]
    # Flows along the edges (range [0, 1])
    flows: List[float]
    # Result of the optimization
    result: MathematicalProgramResult


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
    # Set to empty list to override default costs to be no cost
    costs: List[Cost] = None
    # Set to empty list to override default constraints to be no constraint
    constraints: List[Constraint] = None
    # This will be overwritten when adding vertex to graph
    gcs_vertex: Optional[GraphOfConvexSets.Vertex] = None


@dataclass
class Edge:
    # Source/"left" vertex of the edge
    u: str
    # Target/"right" vertex of the edge
    v: str
    # Set to empty list to override default costs to be no cost
    costs: List[Cost] = None
    # Set to empty list to override default constraints to be no constraint
    constraints: List[Constraint] = None
    # This will be overwritten when adding edge to graph
    gcs_edge: Optional[GraphOfConvexSets.Edge] = None


@dataclass
class GraphParams:
    # Tuple of the smallest and largest ambient dimension of the vertices
    dim_bounds: Tuple[int, int]
    # Number of vertices/convex sets
    n_vertices: int
    # Number of edges
    n_edges: int
    # Source and target coordinates, (centers if they are non singleton sets)
    source: Tuple
    target: Tuple
    # Workspace bounds if specified
    workspace: np.ndarray
    # Default costs and constraints if any
    default_costs_constraints: DefaultGraphCostsConstraints


class Graph:
    """
    Wrapper for Drake GraphOfConvexSets class.
    """

    def __init__(
        self,
        default_costs_constraints: DefaultGraphCostsConstraints = None,
        workspace: np.ndarray = None,
    ) -> None:
        """
        Args:
            default_costs_constraints: Default costs and constraints for vertices and edges.
            workspace_bounds: Bounds on the workspace that will be plotted. Each row should specify lower and upper bounds for a dimension.
        """

        self._default_costs_constraints = default_costs_constraints
        self.vertices = {}
        self.edges = {}
        self._source_name = None
        self._target_name = None

        if workspace is not None:
            assert (
                workspace.shape[1] == 2
            ), "Each row of workspace_bounds should specify lower and upper bounds for a dimension"
            if workspace.shape[0] > 2:
                raise NotImplementedError(
                    "Workspace bounds with more than 2 dimensions not yet supported"
                )
        self.workspace = workspace

        self._gcs = GraphOfConvexSets()

    def add_vertex(self, vertex: Vertex, name: str = ""):
        """
        Add a vertex to the graph.
        """
        if name == "":
            name = len(self.vertices)
        assert name not in self.vertices

        # Makes a copy so that the original vertex is not modified
        # allows for convenient adding of vertices from one graph to another
        v = copy(vertex)

        v.gcs_vertex = self._gcs.AddVertex(v.convex_set.set, name)
        self.vertices[name] = v

        # Set default costs and constraints if necessary
        if self._default_costs_constraints:  # Have defaults
            if (
                v.costs
                is None  # Vertex have not been specifically overriden to be no cost (empty list)
                and self._default_costs_constraints.vertex_costs
            ):
                v.costs = self._default_costs_constraints.vertex_costs
            if (
                v.constraints is None
                and self._default_costs_constraints.vertex_constraints
            ):
                v.constraints = self._default_costs_constraints.vertex_constraints

        # Add costs and constraints to gcs vertex
        if v.costs:
            for cost in v.costs:
                binding = Binding[Cost](cost, v.gcs_vertex.x().flatten())
                v.gcs_vertex.AddCost(binding)
        if v.constraints:
            for constraint in v.constraints:
                binding = Binding[Constraint](constraint, v.gcs_vertex.x().flatten())
                v.gcs_vertex.AddConstraint(binding)

    def remove_vertex(self, name: str):
        """
        Remove a vertex from the graph as well as any edges from or to that vertex.
        """
        self._gcs.RemoveVertex(self.vertices[name].gcs_vertex)
        self.vertices.pop(name)
        for edge in self.edge_keys:
            if name in edge:
                self.remove_edge(
                    edge, remove_from_gcs=False
                )  # gcs.RemoveVertex already removes edges from gcs

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
        e = copy(edge)
        e.gcs_edge = self._gcs.AddEdge(
            self.vertices[e.u].gcs_vertex, self.vertices[e.v].gcs_vertex
        )

        # Set default costs and constraints if necessary
        if self._default_costs_constraints:  # Have defaults
            if (
                e.costs
                is None  # Edge have not been specifically overriden to be no cost (empty list)
                and self._default_costs_constraints.edge_costs
            ):
                e.costs = self._default_costs_constraints.edge_costs
            if (
                e.constraints is None
                and self._default_costs_constraints.edge_constraints
            ):
                e.constraints = self._default_costs_constraints.edge_constraints

        # Add costs and constraints to gcs edge
        if e.costs:
            for cost in e.costs:
                x = np.concatenate([e.gcs_edge.xu(), e.gcs_edge.xv()])
                binding = Binding[Cost](cost, x)
                e.gcs_edge.AddCost(binding)
        if e.constraints:
            for constraint in e.constraints:
                x = np.concatenate([e.gcs_edge.xu(), e.gcs_edge.xv()])
                binding = Binding[Constraint](constraint, x)
                e.gcs_edge.AddConstraint(binding)

        self.edges[(e.u, e.v)] = e

    def remove_edge(self, edge_key: Tuple[str, str], remove_from_gcs: bool = True):
        """
        Remove an edge from the graph.
        """
        if remove_from_gcs:
            self._gcs.RemoveEdge(self.edges[edge_key].gcs_edge)

        self.edges.pop(edge_key)

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
        assert len(us) == len(vs)
        if costs is None:
            costs = [None] * len(us)
        else:
            assert len(costs) == len(us)
        if constraints is None:
            constraints = [None] * len(us)
        else:
            assert len(constraints) == len(us)

        for u, v, cost_list, constraint_list in zip(us, vs, costs, constraints):
            self.add_edge(Edge(u, v, cost_list, constraint_list))

    def set_source(self, vertex_name: str):
        assert vertex_name in self.vertices
        self._source_name = vertex_name

    def set_target(self, vertex_name: str):
        assert vertex_name in self.vertices
        self._target_name = vertex_name

    def outgoing_edges(self, vertex_name: str) -> List[Edge]:
        """
        Get the outgoing edges of a vertex.
        """
        assert vertex_name in self.vertices
        return [edge for edge in self.edges.values() if edge.u == vertex_name]

    def incoming_edges(self, vertex_name: str) -> List[Edge]:
        """
        Get the incoming edges of a vertex.
        """
        assert vertex_name in self.vertices
        return [edge for edge in self.edges.values() if edge.v == vertex_name]

    def incident_edges(self, vertex_name: str) -> List[Edge]:
        """
        Get the incident edges of a vertex.
        """
        assert vertex_name in self.vertices
        return [
            edge
            for edge in self.edges.values()
            if edge.u == vertex_name or edge.v == vertex_name
        ]

    def solve(self, use_convex_relaxation=False) -> ShortestPathSolution:
        """
        Solve the shortest path problem.
        """
        assert self._source_name is not None
        assert self._target_name is not None

        options = GraphOfConvexSetsOptions()

        options.convex_relaxation = use_convex_relaxation
        if use_convex_relaxation is True:
            options.preprocessing = True
            options.max_rounded_paths = 100

        # print(f"target: {self._target_name}, {self.vertices[self._target_name].gcs_vertex}")
        result = self._gcs.SolveShortestPath(
            self.vertices[self._source_name].gcs_vertex,
            self.vertices[self._target_name].gcs_vertex,
            options,
        )
        assert result.is_success()

        sol = self._parse_result(result)

        # Optional post solve hook for subclasses
        self._post_solve(sol)

        return sol

    def _post_solve(self, sol: ShortestPathSolution):
        """Optional post solve hook for subclasses"""
        pass

    def _parse_result(self, result: MathematicalProgramResult) -> ShortestPathSolution:
        cost = result.get_optimal_cost()
        time = result.get_solver_details().optimizer_time

        flow_variables = [e.phi() for e in self._gcs.Edges()]
        flows = [result.GetSolution(p) for p in flow_variables]
        edge_path = []
        for k, flow in enumerate(flows):
            if flow >= 0.99:
                edge_path.append(self.edge_keys[k])
        assert len(self._gcs.Edges()) == self.n_edges

        # Edges are in order they were added to the graph and not in order of the path
        vertex_path = self._convert_active_edges_to_vertex_path(
            self.source_name, self.target_name, edge_path
        )
        # vertex_path = [self.source_name]
        ambient_path = [
            result.GetSolution(self.vertices[v].gcs_vertex.x()) for v in vertex_path
        ]

        return ShortestPathSolution(
            cost, time, vertex_path, ambient_path, flows, result
        )

    @staticmethod
    def _convert_active_edges_to_vertex_path(source_name, target_name, edges):
        # Create a dictionary where the keys are the vertices and the values are their neighbors
        neighbors = {u: v for u, v in edges}
        # Start with the source vertex
        path = [source_name]

        # While the last vertex in the path has a neighbor
        while path[-1] in neighbors:
            # Add the neighbor to the path
            path.append(neighbors[path[-1]])

        assert path[-1] == target_name, "Path does not end at target"

        return path

    def plot_sets(self, **kwargs):
        # Set gridlines to be drawn below the data
        plt.rc("axes", axisbelow=True)
        plt.gca().set_aspect("equal")
        for v in self.vertices.values():
            v.convex_set.plot(**kwargs)

    def plot_edge(self, edge_key, **kwargs):
        options = {
            "color": "k",
            "zorder": 2,
            "arrowstyle": "->, head_width=3, head_length=8",
        }
        options.update(kwargs)
        tail = self.vertices[edge_key[0]].convex_set.center
        head = self.vertices[edge_key[1]].convex_set.center
        arrow = patches.FancyArrowPatch(tail, head, **options)
        plt.gca().add_patch(arrow)

    def plot_edges(self, **kwargs):
        for edge in self.edge_keys:
            self.plot_edge(edge, **kwargs)

    def plot_set_labels(self, labels=None, **kwargs):
        options = {
            "color": "black",
            "zorder": 5,
            "size": 8,
        }
        options.update(kwargs)
        if labels is None:
            labels = self.vertex_names
        offset = np.array([0.2, 0.1])
        patch_offset = np.array([-0.05, -0.1])  # from text
        for v, label in zip(self.vertices.values(), labels):
            pos = v.convex_set.center + offset
            # Create a colored rectangle as the background
            rect = patches.FancyBboxPatch(
                pos + patch_offset,
                0.8,
                0.4,
                color="white",
                alpha=0.8,
                zorder=4,
                boxstyle="round,pad=0.1",
            )
            plt.gca().add_patch(rect)
            plt.text(*pos, label, **options)

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

    def plot_path(self, path: List[np.ndarray], **kwargs):
        options = {
            "color": "g",
            "marker": "o",
            "markeredgecolor": "k",
            "markerfacecolor": "w",
        }
        options.update(kwargs)
        plt.plot(*np.array([x for x in path]).T, **options)

    def graphviz(self):
        vertex_labels = self.vertex_names

        G = Digraph()
        for label in vertex_labels:
            G.node(label)
        for u, v in self.edge_keys:
            G.edge(u, v, "")
        return G

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
        return self._source_name
        # return (
        #     self.vertex_names[self.vertices[0]]
        #     if self._source_name is None
        #     else self._source_name
        # )

    @property
    def target_name(self):
        return self._target_name
        # return (
        #     self.vertex_names[self.vertices[-1]]
        #     if self._target_name is None
        #     else self._target_name
        # )

    @property
    def source(self):
        return self.vertices[self.source_name]

    @property
    def target(self):
        return self.vertices[self.target_name]

    @property
    def dim_bounds(self):
        # Return a tuple of the smallest and largest dimension of the vertices
        smallest_dim = min([v.convex_set.dim for v in self.vertices.values()])
        largest_dim = max([v.convex_set.dim for v in self.vertices.values()])
        return (smallest_dim, largest_dim)

    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def params(self):
        return GraphParams(
            dim_bounds=self.dim_bounds,
            n_vertices=self.n_vertices,
            n_edges=self.n_edges,
            source=self.source.convex_set.center,
            target=self.target.convex_set.center,
            default_costs_constraints=self._default_costs_constraints,
            workspace=self.workspace,
        )
