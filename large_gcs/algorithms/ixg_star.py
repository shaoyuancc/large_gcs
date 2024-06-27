import logging
import time
from typing import Optional

import numpy as np

from large_gcs.algorithms.ixg import IxG
from large_gcs.algorithms.search_algorithm import (
    AlgVisParams,
    SearchNode,
    TieBreak,
    profile_method,
)
from large_gcs.graph.graph import Graph, ShortestPathSolution
from large_gcs.graph.lower_bound_graph import LowerBoundGraph

logger = logging.getLogger(__name__)


class IxGStar(IxG):
    def __init__(
        self,
        graph: Graph,
        lbg: LowerBoundGraph,
        heuristic_inflation_factor: float = 1,
        vis_params: Optional[AlgVisParams] = None,
        tiebreak: TieBreak = TieBreak.FIFO,
    ):
        super().__init__(
            graph=graph,
            lbg=lbg,
            heuristic_inflation_factor=heuristic_inflation_factor,
            vis_params=vis_params,
            tiebreak=tiebreak,
        )
        # Initialization is exactly the same except we also initialize IxG
        self._ixg = IxG(
            graph,
            lbg,
            heuristic_inflation_factor,
            vis_params=None,
            should_save_metrics=False,
        )

    def run(self):
        logger.info(f"Running {self.__class__.__name__}")
        sol: Optional[ShortestPathSolution] = None
        start_time = time.time()

        ixg_sol = self._ixg.run()
        if ixg_sol is not None and ixg_sol.is_success:
            self._ub = ixg_sol.cost * self._heuristic_inflation_factor
        else:
            self._ub = np.inf

        # LBG already updated with source and target by IxG
        # Dijkstra already run by IxG

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
        # Call post-solve again in case other solutions were found after this was first visited.
        self._graph._post_solve(sol)
        return sol

    @profile_method
    def _run_iteration(self) -> Optional[ShortestPathSolution]:
        # Get the node with the lowest cost to come
        n: SearchNode = self.pop_node_from_Q()

        # Check termination condition
        if n.vertex_name == self._target_name:
            self._save_metrics(n, [], override_save=True)
            return n.sol

        self.update_expanded(n)

        edges = self._graph.outgoing_edges(n.vertex_name)
        self._save_metrics(n, edges)
        for edge in edges:
            self._explore_successor(n, edge)

    @profile_method
    def _explore_successor(self, n: SearchNode, edge) -> None:
        n_next = SearchNode.from_parent(child_vertex_name=edge.v, parent=n)
        self._graph.set_target(n_next.vertex_name)
        sol = self._graph.solve_convex_restriction(
            active_edge_keys=n_next.edge_path,
            skip_post_solve=False,
            # skip_post_solve=(not n_next.vertex_name == self._target_name),
        )
        if not sol.is_success:
            logger.debug(f"Path not actually feasible")
            # Path invalid, do nothing, don't add to Q
            return
        else:
            logger.debug(f"Path is feasible")

        lb = sol.cost + self._heuristic_inflation_factor * self._lbg.get_cost_to_go(
            n_next.vertex_name
        )

        if lb < self._ub:
            self._g[n_next.vertex_name] = sol.cost
            n_next.sol = sol
            # Get priority from LBG
            n_next.priority = lb
            self.push_node_on_Q(n_next)
            self.add_node_to_S(n_next)
            self.update_visited(n_next)
        else:
            self.update_pruned(n_next)
