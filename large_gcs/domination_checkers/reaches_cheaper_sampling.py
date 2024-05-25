import logging
from typing import List, Tuple

import numpy as np

from large_gcs.algorithms.search_algorithm import SearchNode
from large_gcs.domination_checkers.sampling_domination_checker import (
    SamplingDominationChecker,
    SetSamples,
)
from large_gcs.geometry.point import Point
from large_gcs.graph.graph import Edge, ShortestPathSolution, Vertex

logger = logging.getLogger(__name__)


class ReachesCheaperSampling(SamplingDominationChecker):
    """Checks samples to see if this path reaches any projected sample cheaper
    than any previous path.

    Assumes that this path is feasible.
    """

    def _is_single_dominated(
        self, candidate_sol: ShortestPathSolution, alt_sol: ShortestPathSolution
    ) -> bool:
        # Assumes candidate_sol is feasible
        return alt_sol.is_success and alt_sol.cost <= candidate_sol.cost

    def _compute_candidate_sol(
        self, candidate_node: SearchNode, sample_vertex_name: str
    ) -> Tuple[ShortestPathSolution | None, bool]:
        candidate_sol = self._solve_conv_res_to_sample(
            candidate_node, sample_vertex_name
        )
        if not candidate_sol.is_success:
            logger.error(
                f"Candidate path was not feasible to reach {sample_vertex_name}"
                f"\nvertex_path: {candidate_node.vertex_path}"
                f"\n Skipping to next sample"
            )
            # assert sol.is_success, "Candidate path should be feasible"

        return candidate_sol, candidate_sol.is_success
