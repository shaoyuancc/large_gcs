import gc
import itertools
import logging
import os
import pickle
import time
from collections import defaultdict, deque
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from large_gcs.algorithms.search_algorithm import SearchAlgorithm, SearchNode
from large_gcs.graph.graph import Graph, ShortestPathSolution
from large_gcs.graph.lower_bound_graph import LowerBoundGraph

logger = logging.getLogger(__name__)


class IxG(SearchAlgorithm):
    def __init__(
        self,
        graph: Graph,
        eps: float = 1,
    ):
        self._graph = graph
        self._eps = eps
        # Stores the search node with the lowest cost to come found so far for each vertex
        self._S = dict[str, SearchNode] = {}
        # Stores the cost to come found so far for each vertex
        self._g = dict[str, float] = defaultdict(lambda: np.inf)
        self._expanded = set()
        # Priority queue
        self._Q = []

        start_node = SearchNode(
            priority=0,
            vertex_name=self._graph.source_name,
            edge_path=[],
            vertex_path=[self._graph.source_name],
            sol=None,
        )
        self._g[self._graph.source_name] = 0
        self.push_node_on_Q(start_node)

    def run(self):
        logger.info(f"Running {self.__class__.__name__}")
        sol: Optional[ShortestPathSolution] = None
        start_time = time.time()

        self.create_lbg()

        while sol == None and len(self._Q) > 0:
            sol = self._run_iteration()
            self._alg_metrics.time_wall_clock = time.time() - start_time
        if sol is None:
            logger.warning(
                f"{self.__class__.__name__} failed to find a path to the target."
            )
            return
        logger.info(
            f"{self.__class__.__name__} complete! \ncost: {sol.cost}, time: {sol.time}"
            f"\nvertex path: {np.array(sol.vertex_path)}"
        )
        return sol

    def _run_iteration(self) -> Optional[ShortestPathSolution]:
        # Get the node with the lowest cost to come
        n: SearchNode = self.pop_node_from_Q()

        # Check termination condition
        if n.vertex_name == self._graph.target_name:
            # self._save_metrics(n, [], override_save=True)
            return n.sol

        # Only expand the node if it has not been expanded before
        if n.vertex_name in self._expanded:
            return None
        self._expanded.add(n.vertex_name)

        edges = self._graph.outgoing_edges(n.vertex_name)
        for edge in edges:
            # Not part of IxG, should replace solve_convex_restriction with something that can handle repeated vertices
            neighbor_in_path = any(
                (self._graph.edges[e].u == edge.v or self._graph.edges[e].v == edge.v)
                for e in n.edge_path
            )
            if neighbor_in_path:
                continue

            self._visit_neighbor(n, edge)

    def _visit_neighbor(self, n: SearchNode, edge) -> None:
        n_next = SearchNode.from_parent(child_vertex_name=edge.v, parent=n)

        sol = self._graph.solve_convex_restriction(
            active_edge_keys=n_next.edge_path, skip_post_solve=True
        )
        if not sol.is_success:
            return

        if sol.cost < self._g[n_next.vertex_name]:
            self._g[n_next.vertex_name] = sol.cost
            n_next.sol = sol
            # Get priority from LBG
            n_next.priority = sol.cost + self._eps * self._lbg.get_cost_to_go(
                n_next.vertex_name
            )
            self.push_node_on_Q(n_next)

    def create_lbg(self):
        self._lbg = LowerBoundGraph(self._graph)
