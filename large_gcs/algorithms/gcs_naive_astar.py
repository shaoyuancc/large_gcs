import logging
import time
from typing import List, Optional

import numpy as np

from large_gcs.algorithms.search_algorithm import (
    AlgVisParams,
    ReexploreLevel,
    SearchAlgorithm,
    SearchNode,
    TieBreak,
    profile_method,
)
from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.graph.graph import Graph, ShortestPathSolution
from large_gcs.graph.incremental_contact_graph import IncrementalContactGraph

logger = logging.getLogger(__name__)


class GcsNaiveAstar(SearchAlgorithm):
    """Naive A* applied to GCS, where in the subroutine, the order of vertices
    is fixed."""

    def __init__(
        self,
        graph: Graph,
        cost_estimator: CostEstimator,
        reexplore_level: ReexploreLevel = ReexploreLevel.PARTIAL,
        tiebreak: TieBreak = TieBreak.FIFO,
        vis_params: AlgVisParams = AlgVisParams(),
        heuristic_inflation_factor: float = 1,
        terminate_early: bool = False,
        allow_cycles: bool = True,
    ):
        if isinstance(graph, IncrementalContactGraph):
            assert (
                graph._should_add_gcs == True
            ), "Required because operating directly on graph instead of subgraph"
        super().__init__(
            graph=graph,
            heuristic_inflation_factor=heuristic_inflation_factor,
            vis_params=vis_params,
            tiebreak=tiebreak,
        )
        self._cost_estimator = cost_estimator
        self._terminate_early = terminate_early
        self._reexplore_level = (
            ReexploreLevel[reexplore_level]
            if type(reexplore_level) == str
            else reexplore_level
        )
        self._cost_estimator.set_alg_metrics(self._alg_metrics)
        self._allow_cycles = allow_cycles

        # For logging/metrics
        call_structure = {
            "_run_iteration": [
                "_explore_successor",
                "_generate_successors",
                "_save_metrics",
            ],
        }
        self._alg_metrics.update_method_call_structure(call_structure)

        start_node = SearchNode.from_source(self._graph.source_name)
        self.push_node_on_Q(start_node)

    def run(
        self,
        visualize_intermediate: bool = False,
        intermediate_vertices_to_visualize: Optional[List[str]] = None,
    ):
        logger.info(
            f"Running {self.__class__.__name__}, reexplore_level: {self._reexplore_level}"
        )
        self._visualize_intermediate = visualize_intermediate
        self._intermediate_sets_to_visualize = intermediate_vertices_to_visualize

        start_time = time.time()
        sol: Optional[ShortestPathSolution] = None
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

    def _run_iteration(self) -> Optional[ShortestPathSolution]:
        n: SearchNode = self.pop_node_from_Q()

        if (
            self._reexplore_level == ReexploreLevel.NONE
            and n.vertex_name in self._expanded
        ):
            return

        # Check termination condition
        if n.vertex_name == self._graph.target_name:
            self._save_metrics(n, [], override_save=True)
            self._maybe_plot_search_node_and_graph(n, is_final_path=True)
            return n.sol

        if n.vertex_name not in self._expanded:
            # Generate successors that you are about to explore
            self._generate_successors(n.vertex_name)
        self.update_expanded(n)

        successors = self._graph.successors(n.vertex_name)

        self._save_metrics(n, len(successors))
        if self._visualize_intermediate:
            self._maybe_plot_search_node_and_graph(n, is_final_path=False)

        for v in successors:
            if not self._allow_cycles and v in n.vertex_path:
                continue

            early_terminate_sol = self._explore_successor(n, v)
            if early_terminate_sol is not None:
                return early_terminate_sol

    def _maybe_plot_search_node_and_graph(self, n: SearchNode, is_final_path: bool):
        if self._visualize_intermediate and (
            self._intermediate_sets_to_visualize is None
            or n in self._intermediate_sets_to_visualize
        ):

            self.plot_search_node_and_graph(n, is_final_path=is_final_path)

    @profile_method
    def _generate_successors(self, vertex_name: str) -> None:
        """Generates neighbors for the given vertex.

        Wrapped to allow for profiling.
        """
        self._graph.generate_successors(vertex_name)

    profile_method

    def _explore_successor(
        self, n: SearchNode, successor: str
    ) -> Optional[ShortestPathSolution]:

        sol: ShortestPathSolution = self._cost_estimator.estimate_cost(
            self._graph,
            successor,
            n,
            heuristic_inflation_factor=self._heuristic_inflation_factor,
            solve_convex_restriction=True,
            override_skip_post_solve=False if self._visualize_intermediate else None,
        )

        if not sol.is_success:
            logger.debug(f"Path not actually feasible")
            # Path invalid, do nothing, don't add to Q
            return
        else:
            logger.debug(f"Path is feasible")

        n_next = SearchNode.from_parent(child_vertex_name=successor, parent=n)
        n_next.sol = sol
        n_next.priority = sol.cost
        logger.debug(
            f"Exploring path (length {len(n_next.vertex_path)}) {n_next.vertex_path}"
        )

        # Here we compare total-estimated-cost \Tilde{f} instead of cost-to-come \Tilde{g}
        # as presented in alg 1. in the paper.
        # For discrete graphs comparing \Tilde{g} is equivalent to comparing \Tilde{f}
        # since the heuristic cost-to-go is the same for a particular terminal vertex.
        # On GCS, they are not equivalent, but we implement this version since we already
        # calculate \Tilde{f} and \Tilde{g} is not currently exposed by Drake when
        # solving the convex restriction.
        # See https://github.com/RobotLocomotion/drake/issues/20443
        if successor != self._target_name and not self._should_add_to_pq(
            successor, sol.cost
        ):
            logger.debug(f"Not added to Q: Path to is dominated")
            self.update_pruned(n_next)
            return
        logger.debug(f"Added to Q: Path not dominated")
        self.set_node_in_S(n_next)
        self.push_node_on_Q(n_next)
        self.update_visited(n_next)

        # Early Termination
        if self._terminate_early and successor == self._target_name:
            logger.info(f"EARLY TERMINATION: Visited path to target.")
            self._save_metrics(n_next, [], override_save=True)
            self._maybe_plot_search_node_and_graph(n_next, is_final_path=True)
            return n_next.sol

    def _should_add_to_pq(self, successor, new_cost):
        if self._reexplore_level == ReexploreLevel.FULL:
            return True
        elif self._reexplore_level == ReexploreLevel.PARTIAL:
            return (successor not in self._S) or (
                new_cost < self._S[successor][0].priority
            )
        elif self._reexplore_level == ReexploreLevel.NONE:
            return successor not in self._visited
