import itertools
import logging
import time
from typing import Optional, Tuple

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
from large_gcs.graph.graph import Edge, Graph, ShortestPathSolution

logger = logging.getLogger(__name__)


class GcsAstarReachability(SearchAlgorithm):
    """
    Reachability based satisficing search on a graph of convex sets using Best-First Search.
    """

    def __init__(
        self,
        graph: Graph,
        cost_estimator: CostEstimator,
        tiebreak: TieBreak = TieBreak.NONE,
        vis_params: Optional[AlgVisParams] = None,
    ):
        super().__init__()
        self._graph = graph
        self._target = graph.target_name
        self._cost_estimator = cost_estimator
        self.vis_params = vis_params
        self._cost_estimator.set_alg_metrics(self._alg_metrics)
        if tiebreak == TieBreak.FIFO or tiebreak == TieBreak.FIFO.name:
            self._counter = itertools.count(start=0, step=1)
        elif tiebreak == TieBreak.LIFO or tiebreak == TieBreak.LIFO.name:
            self._counter = itertools.count(start=0, step=-1)

        self._S: dict[str, list[SearchNode]] = {}  # Expanded/Closed set
        self._Q: list[SearchNode] = []  # Priority queue

        start_node = SearchNode(
            priority=0, vertex_name=self._graph.source_name, path=[], sol=None
        )
        self.push_node_on_Q(start_node)

        # Initialize the "expanded" subgraph on which
        # all the convex restrictions will be solved.
        self._subgraph = Graph(self._graph._default_costs_constraints)
        # Accounting of the full dimensional vertices in the visited subgraph
        self._subgraph_fd_vertices = set()

        # Shouldn't need to add the target since that should be handled by _set_subgraph
        # # Add the target to the visited subgraph
        # self._subgraph.add_vertex(
        #     self._graph.vertices[self._graph.target_name], self._graph.target_name
        # )
        # self._subgraph.set_target(self._graph.target_name)
        # # Start with the target node in the visited subgraph
        # self.alg_metrics.n_vertices_expanded[0] = 1
        # self._subgraph_fd_vertices.add(self._graph.target_name)

        self._cost_estimator.setup_subgraph(self._subgraph)

        self._set_samples

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
        self._set_subgraph(n)

        edges = self._graph.outgoing_edges(n.vertex_name)
        for edge in edges:
            if self._reaches_new(n, edge):
                self._visit_neighbor(n, edge)

    def _reaches_new(
        self, n: SearchNode, edge: Edge
    ) -> Tuple[bool, ShortestPathSolution]:

        sol: ShortestPathSolution = self._cost_estimator.estimate_cost(
            self._subgraph,
            edge,
            n.path,
            solve_convex_restriction=True,
        )
        if not sol.is_success:
            return False, sol

        # Path is feasible, check if it reaches new areas within the set

    def _visit_neighbor(self, n: SearchNode, edge: Edge) -> None:
        neighbor = edge.v
        if neighbor in self._S:
            self._alg_metrics.n_vertices_revisited[0] += 1
        else:
            self._alg_metrics.n_vertices_visited[0] += 1
        # logger.info(f"exploring edge {edge.u} -> {edge.v}")
        sol: ShortestPathSolution = self._cost_estimator.estimate_cost(
            self._subgraph,
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
        n_next.sol = sol
        n_next.priority = sol.cost
        self.push_node_on_Q(n_next)

    def _set_subgraph(self, n: SearchNode) -> None:
        """
        Set the subgraph to contain only the vertices and edges in the
        path of the given search node.
        """
        vertices_to_add = set(
            [self._graph.target_name, self._graph.source_name, n.vertex_name]
        )
        for _, v in n.path:
            vertices_to_add.add(v)

        # Ignore cfree subgraph sets,
        # Remove full dimensional sets if they aren't in the path
        # Add all vertices that aren't already inside
        for v in self._subgraph_fd_vertices.copy():
            if v not in vertices_to_add:  # We don't want to have it so remove it
                self._subgraph.remove_vertex(v)
                self._subgraph_fd_vertices.remove(v)
            else:  # We do want to have it but it's already in so don't need to add it
                vertices_to_add.remove(v)

        for v in vertices_to_add:
            self._subgraph.add_vertex(self._graph.vertices[v], v)
            self._subgraph_fd_vertices.add(v)

        self._subgraph.set_source(self._graph.source_name)
        self._subgraph.set_target(self._graph.target_name)

        # Add edges that aren't already in the visited subgraph.
        for edge_key in n.path:
            if edge_key not in self._subgraph.edges:
                self._subgraph.add_edge(self._graph.edges[edge_key])
