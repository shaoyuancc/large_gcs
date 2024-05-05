import logging
from typing import List

from large_gcs.algorithms.search_algorithm import SearchNode
from large_gcs.domination_checkers.ah_containment_domination_checker import (
    AHContainmentDominationChecker,
)

logger = logging.getLogger(__name__)


class ReachesCheaperContainment(AHContainmentDominationChecker):
    def is_dominated(
        self, candidate_node: SearchNode, alternate_nodes: List[SearchNode]
    ) -> bool:
        """
        Checks if a candidate path is dominated completely by any one of the alternate paths.
        """
        A_n, b_n = self.get_epigraph_matrices(candidate_node)
        T_n = self.get_projection_transformation(
            node=candidate_node,
            A=A_n,
            include_cost_epigraph=True,
        )
        for alt_n in alternate_nodes:
            A_alt, b_alt = self.get_epigraph_matrices(alt_n)
            T_alt = self.get_projection_transformation(
                node=alt_n,
                A=A_alt,
                include_cost_epigraph=True,
            )
            if self.is_contained_in(A_n, b_n, T_n, A_alt, b_alt, T_alt):
                return True
        return False
