import logging
import time
from collections import defaultdict
from typing import List, Optional

import numpy as np

from large_gcs.algorithms.search_algorithm import (
    AlgVisParams,
    SearchAlgorithm,
    SearchNode,
    TieBreak,
    profile_method,
)
from large_gcs.graph.graph import Graph, ShortestPathSolution
from large_gcs.graph.lower_bound_graph import LowerBoundGraph

logger = logging.getLogger(__name__)


class IxG(SearchAlgorithm):
    def __init__(
        self,
        graph: Graph,
        lbg: LowerBoundGraph,
        heuristic_inflation_factor: float = 1,
        vis_params: Optional[AlgVisParams] = None,
        tiebreak: TieBreak = TieBreak.FIFO,
        should_save_metrics: bool = True,
    ):
        super().__init__(
            graph=graph,
            heuristic_inflation_factor=heuristic_inflation_factor,
            vis_params=vis_params,
            tiebreak=tiebreak,
        )
        # Override whether to save metrics so that when IxG* calls IxG, it doesn't save metrics.
        self._should_save_metrics = should_save_metrics
        # Stores the cost to come found so far for each vertex
        self._g: dict[str, float] = defaultdict(lambda: np.inf)
        # Store a reference to Lower Bound Graph
        self._lbg = lbg
        # Set the graph used by the Lower Bound Graph
        self._lbg._graph = graph

        # For logging/metrics
        call_structure = {
            "_run_iteration": [
                "_explore_successor",
                "_save_metrics",
            ]
        }
        self._alg_metrics.update_method_call_structure(call_structure)

        start_node = SearchNode.from_source(self._graph.source_name)
        self._g[self._graph.source_name] = 0
        self.push_node_on_Q(start_node)

    def run(self):
        logger.info(f"Running {self.__class__.__name__}")
        sol: Optional[ShortestPathSolution] = None
        start_time = time.time()

        # Update lbg with source and target
        self._lbg.update_lbg(self._graph.source_name, self._graph.source)
        self._lbg.update_lbg(self._target_name, self._graph.target)
        update_duration = time.time() - start_time
        logger.info(
            f"Updated LBG with source and target in {update_duration:.2f} seconds"
        )

        self._lbg.run_dijkstra(self._graph.target_name)

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

    @profile_method
    def _run_iteration(self) -> Optional[ShortestPathSolution]:
        # Get the node with the lowest cost to come
        n: SearchNode = self.pop_node_from_Q()
        # import pdb; pdb.set_trace()
        # Only expand the node if it has not been expanded before
        if n.vertex_name in self._expanded:
            return None
        self.update_expanded(n)

        edges = self._graph.outgoing_edges(n.vertex_name)
        if self._should_save_metrics:
            self._save_metrics(n, edges)
        for edge in edges:
            # Check early termination condition
            sol = self._explore_successor(n, edge)
            if sol is not None:
                return sol

    @profile_method
    def _explore_successor(self, n: SearchNode, edge) -> None:
        n_next = SearchNode.from_parent(child_vertex_name=edge.v, parent=n)
        self._graph.set_target(n_next.vertex_name)
        sol = self._graph.solve_convex_restriction(
            active_edge_keys=n_next.edge_path,
            skip_post_solve=(not n_next.vertex_name == self._target_name),
        )
        if not sol.is_success:
            logger.debug(f"Path not actually feasible")
            # Path invalid, do nothing, don't add to Q
            return
        else:
            logger.debug(f"Path is feasible")

        if sol.cost < self._g[n_next.vertex_name]:
            self._g[n_next.vertex_name] = sol.cost
            n_next.sol = sol
            # Get priority from LBG
            n_next.priority = (
                sol.cost
                + self._heuristic_inflation_factor
                * self._lbg.get_cost_to_go(n_next.vertex_name)
            )
            self.push_node_on_Q(n_next)
            self.set_node_in_S(n_next)
            self.update_visited(n_next)

            if n_next.vertex_name == self._target_name:
                if self._should_save_metrics:
                    self._save_metrics(n_next, [], override_save=True)
                return n_next.sol
        else:
            self.update_pruned(n_next)

    @profile_method
    def _solve_convex_restriction(
        self, active_edge_keys: List[str], skip_post_solve: bool = False
    ):
        sol = self._graph.solve_convex_restriction(active_edge_keys, skip_post_solve)
        self._alg_metrics.update_after_gcs_solve(sol.time)
        return sol
