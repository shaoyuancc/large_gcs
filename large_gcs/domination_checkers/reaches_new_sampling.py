import logging
from typing import Tuple

from large_gcs.algorithms.search_algorithm import SearchNode
from large_gcs.domination_checkers.sampling_domination_checker import (
    SamplingDominationChecker,
)
from large_gcs.graph.graph import ShortestPathSolution

logger = logging.getLogger(__name__)


class ReachesNewSampling(SamplingDominationChecker):
    """Checks samples to see if this path reaches new previously unreached
    samples.

    Assumes that this path is feasible.
    """

    def _is_single_dominated(
        self, candidate_sol: ShortestPathSolution, alt_sol: ShortestPathSolution
    ) -> bool:
        # For reaches new, as long as the alt sol is feasible, it dominates the candidate
        return alt_sol.is_success

    def _compute_candidate_sol(
        self, candidate_node: SearchNode, sample_vertex_name: str
    ) -> Tuple[ShortestPathSolution | None, bool]:
        # For reaches new, we don't need to compute the candidate solution,
        # We assume that the projection step led to a feasible solution for the candidate
        return None, True
