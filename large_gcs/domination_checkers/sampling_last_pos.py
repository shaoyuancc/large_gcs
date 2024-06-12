import logging
from typing import Tuple

from numpy import ndarray

from large_gcs.algorithms.search_algorithm import SearchNode
from large_gcs.contact.contact_set import ContactSet
from large_gcs.domination_checkers.sampling_domination_checker import (
    SamplingDominationChecker,
)
from large_gcs.graph.contact_cost_constraint_factory import (
    vertex_constraint_last_pos_equality,
)
from large_gcs.graph.graph import ShortestPathSolution, Vertex

logger = logging.getLogger(__name__)


class SamplingLastPos(SamplingDominationChecker):
    """Checks samples to see if this path reaches new previously unreached
    samples.

    Assumes that this path is feasible.
    """

    def _add_sample_to_graph(
        self, sample: ndarray, sample_vertex_name: str, candidate_node: SearchNode
    ) -> None:
        # logger.debug(f"_add_sample_to_graph in SamplingLastPos")
        contact_set: ContactSet = self._graph.vertices[
            candidate_node.vertex_name
        ].convex_set
        # logger.debug(f"sample: {sample}")
        self._graph.add_vertex(
            vertex=Vertex(
                convex_set=contact_set,
                constraints=[
                    vertex_constraint_last_pos_equality(contact_set.vars, sample)
                ],
            ),
            name=sample_vertex_name,
        )


class ReachesNewLastPosSampling(SamplingLastPos):
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


class ReachesCheaperLastPosSampling(SamplingLastPos):
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
        return candidate_sol, candidate_sol.is_success
