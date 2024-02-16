import itertools
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib.animation import FFMpegWriter

from large_gcs.algorithms.search_algorithm import (
    AlgMetrics,
    AlgVisParams,
    SearchAlgorithm,
    SearchNode,
    TieBreak,
)
from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.geometry.convex_set import ConvexSet
from large_gcs.geometry.point import Point
from large_gcs.graph.cost_constraint_factory import create_equality_edge_constraint
from large_gcs.graph.graph import Edge, Graph, ShortestPathSolution, Vertex
from large_gcs.graph.incremental_contact_graph import IncrementalContactGraph

logger = logging.getLogger(__name__)


@dataclass
class SetSample:
    vertex_name: str
    index: int
    convex_set: ConvexSet
    reached_by: Optional[SearchNode] = None

    @property
    def id(self):
        return f"{self.vertex_name}_sample_{self.index}"


@dataclass
class SetSamples:
    vertex_name: str
    unreached_samples: List[SetSample]
    reached_samples: List[SetSample]

    @property
    def has_unreached_samples(self):
        return len(self.unreached_samples) > 0

    @classmethod
    def from_vertex(cls, vertex_name: str, vertex: Vertex, num_samples: int):
        samples = vertex.convex_set.get_samples(num_samples)
        return cls(
            vertex_name=vertex_name,
            unreached_samples=[
                SetSample(vertex_name=vertex_name, index=i, convex_set=Point(x))
                for i, x in enumerate(samples)
            ],
            reached_samples=[],
        )

    def add_samples_to_graph(self, graph: Graph):
        parent_vertex = graph.vertices[self.vertex_name]
        for sample in self.unreached_samples:
            graph.add_vertex(
                vertex=Vertex(convex_set=sample.convex_set), name=sample.id
            )
            graph.add_edge(
                Edge(
                    u=self.vertex_name,
                    v=sample.id,
                    costs=None,
                    constraints=[
                        create_equality_edge_constraint(
                            dim=parent_vertex.convex_set.dim
                        )
                    ],
                )
            )


class GcsAstarReachability(SearchAlgorithm):
    """
    Reachability based satisficing search on a graph of convex sets using Best-First Search.
    Note:
    - Use with factored_collision_free cost estimator not yet implemented.
    In particular, this doesn't use a subgraph, but operates directly on the graph.
    """

    def __init__(
        self,
        graph: Graph,
        cost_estimator: CostEstimator,
        tiebreak: TieBreak = TieBreak.FIFO,
        vis_params: Optional[AlgVisParams] = None,
        num_samples_per_vertex: int = 100,
    ):
        if isinstance(graph, IncrementalContactGraph):
            assert (
                graph._should_add_gcs == True
            ), "Required because operating directly on graph instead of subgraph"
        super().__init__()
        self._graph = graph
        self._target = graph.target_name
        self._cost_estimator = cost_estimator
        self._vis_params = vis_params
        self._num_samples_per_vertex = num_samples_per_vertex
        self._cost_estimator.set_alg_metrics(self._alg_metrics)
        if tiebreak == TieBreak.FIFO or tiebreak == TieBreak.FIFO.name:
            self._counter = itertools.count(start=0, step=1)
        elif tiebreak == TieBreak.LIFO or tiebreak == TieBreak.LIFO.name:
            self._counter = itertools.count(start=0, step=-1)

        self._S: dict[str, list[SearchNode]] = {}  # Expanded/Closed set
        self._Q: list[SearchNode] = []  # Priority queue
        # Keeps track of samples for each vertex(set) in the graph for which the vertex has been visited but the sample has not been reached.
        self._set_samples: dict[str, SetSamples] = {}

        start_node = SearchNode(
            priority=0, vertex_name=self._graph.source_name, path=[], sol=None
        )
        self.push_node_on_Q(start_node)

        # Shouldn't need the subgraph, just operate directly on the graph
        # # Initialize the "expanded" subgraph on which
        # # all the convex restrictions will be solved.
        # self._subgraph = Graph(self._graph._default_costs_constraints)
        # # Accounting of the full dimensional vertices in the visited subgraph
        # self._subgraph_fd_vertices = set()

        # Shouldn't need to add the target since that should be handled by _set_subgraph
        # # Add the target to the visited subgraph
        # self._subgraph.add_vertex(
        #     self._graph.vertices[self._graph.target_name], self._graph.target_name
        # )
        # self._subgraph.set_target(self._graph.target_name)
        # # Start with the target node in the visited subgraph
        # self.alg_metrics.n_vertices_expanded[0] = 1
        # self._subgraph_fd_vertices.add(self._graph.target_name)

        self._cost_estimator.setup_subgraph(self._graph)

    def run(self) -> ShortestPathSolution:
        """
        Searches for a shortest path in the given graph.
        """
        logger.info(f"Running {self.__class__.__name__}")
        start_time = time.time()
        sol: Optional[ShortestPathSolution] = None
        while sol == None and len(self._Q) > 0:
            sol = self._run_iteration()
            self._alg_metrics.time_wall_clock = time.time() - start_time
        if sol is None:
            logger.warn(
                f"{self.__class__.__name__} failed to find a path to the target."
            )
            return

        self._graph._post_solve(sol)
        logger.info(
            f"{self.__class__.__name__} complete! \ncost: {sol.cost}, time: {sol.time}"
            f"\nvertex path: {np.array(sol.vertex_path)}"
            f"\n{self.alg_metrics}"
        )
        return sol

    def _run_iteration(self) -> Optional[ShortestPathSolution]:
        """
        Runs one iteration of the search algorithm.
        """
        n: SearchNode = self.pop_node_from_Q()

        # Check termination condition
        if n.vertex_name == self._graph.target_name:
            return n.sol

        # TODO: update alg metrics

        # Generate neighbors that you are about to explore/visit
        self._graph.generate_neighbors(n.vertex_name)

        # Configure subgraph for visiting neighbors
        # Don't need this because operating directly on the graph
        # self._set_subgraph(n)

        edges = self._graph.outgoing_edges(n.vertex_name)
        for edge in edges:
            if "sample" not in edge.v:
                self._visit_neighbor(n, edge)

    def _visit_neighbor(self, n: SearchNode, edge: Edge) -> None:
        neighbor = edge.v
        if neighbor in self._S:
            self._alg_metrics.n_vertices_revisited[0] += 1
        else:
            self._alg_metrics.n_vertices_visited[0] += 1
        logger.info(f"exploring edge {edge.u} -> {edge.v}")
        sol: ShortestPathSolution = self._cost_estimator.estimate_cost_on_graph(
            self._graph,
            edge,
            n.path,
            solve_convex_restriction=True,
        )

        if not sol.is_success:
            logger.debug(f"edge {n.vertex_name} -> {edge.v} not actually feasible")
            # Path invalid, do nothing, don't add to Q
            return
        else:
            logger.debug(f"edge {n.vertex_name} -> {edge.v} is feasible")

        n_next = SearchNode.from_parent(child_vertex_name=edge.v, parent=n)

        if not self._reaches_new(n_next):
            return

        n_next.sol = sol
        n_next.priority = sol.cost
        self.push_node_on_Q(n_next)

    def _reaches_new(self, n_next: SearchNode) -> bool:
        """
        Checks samples to see if this path reaches new previously unreached samples.
        Assumes that this path is feasible.
        """

        if n_next.vertex_name not in self._set_samples:
            self._set_samples[n_next.vertex_name] = SetSamples.from_vertex(
                n_next.vertex_name,
                self._graph.vertices[n_next.vertex_name],
                self._num_samples_per_vertex,
            )
            self._set_samples[n_next.vertex_name].add_samples_to_graph(self._graph)

        set_samples = self._set_samples[n_next.vertex_name]
        still_unreached = []
        reached_new = False
        for sample in set_samples.unreached_samples:
            # Check whether sample can be reached via the path
            self._graph.set_target(sample.id)
            conv_res_active_edges = n_next.path.copy() + [
                (n_next.vertex_name, sample.id)
            ]
            logger.debug(f"reaches new, active edges: {conv_res_active_edges}")
            sol = self._graph.solve_convex_restriction(conv_res_active_edges)
            self._graph.set_target(self._target)
            if sol.is_success:
                sample.reached_by = n_next
                set_samples.reached_samples.append(sample)
                reached_new = True
            else:
                still_unreached.append(sample)

        set_samples.unreached_samples = still_unreached
        logger.debug(f"{n_next.vertex_name} Reached new samples: {reached_new}")
        return reached_new


"""
Actually I think we don't need to operate on this subgraph at all but just the whole graph.
The vertices and edges that are not in the path are going to be cleared by the C++ code,
so we don't need to worry about that.
"""
# def _set_subgraph(self, n: SearchNode) -> None:
#     """
#     Set the subgraph to contain only the vertices and edges in the
#     path of the given search node.
#     """
#     vertices_to_add = set(
#         [self._graph.target_name, self._graph.source_name, n.vertex_name]
#     )
#     for _, v in n.path:
#         vertices_to_add.add(v)

#     # Ignore cfree subgraph sets,
#     # Remove full dimensional sets if they aren't in the path
#     # Add all vertices that aren't already inside
#     for v in self._subgraph_fd_vertices.copy():
#         if v not in vertices_to_add:  # We don't want to have it so remove it
#             self._subgraph.remove_vertex(v)
#             self._subgraph_fd_vertices.remove(v)
#         else:  # We do want to have it but it's already in so don't need to add it
#             vertices_to_add.remove(v)

#     for v in vertices_to_add:
#         self._subgraph.add_vertex(self._graph.vertices[v], v)
#         self._subgraph_fd_vertices.add(v)

#     self._subgraph.set_source(self._graph.source_name)
#     self._subgraph.set_target(self._graph.target_name)

#     # Add edges that aren't already in the visited subgraph.
#     for edge_key in n.path:
#         if edge_key not in self._subgraph.edges:
#             self._subgraph.add_edge(self._graph.edges[edge_key])
