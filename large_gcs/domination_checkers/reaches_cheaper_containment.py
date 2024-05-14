import logging
from typing import List

from pydrake.all import HPolyhedron, VPolytope

from large_gcs.algorithms.search_algorithm import SearchNode
from large_gcs.domination_checkers.ah_containment_domination_checker import (
    AHContainmentDominationChecker,
)
from large_gcs.geometry.polyhedron import Polyhedron

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

    def plot_containment(
        self,
        candidate_node: SearchNode,
        alternate_nodes: List[SearchNode],
        cost_upper_bound: float = 20,
    ):
        A_n, b_n = self.get_epigraph_matrices(
            candidate_node, add_upper_bound=True, cost_upper_bound=cost_upper_bound
        )
        T_n = self.get_projection_transformation(
            node=candidate_node,
            A=A_n,
            include_cost_epigraph=True,
        )
        P_n = Polyhedron(A_n, b_n)
        P_n._vertices = VPolytope(HPolyhedron(A_n, b_n)).vertices().T
        fig = P_n.plot_transformation(
            T_n,
            opacity=0.3,
            color="blue",
            name=f"Candidate:{candidate_node.vertex_path}",
            showlegend=True,
        )
        for alternate_node in alternate_nodes:
            A_alt, b_alt = self.get_epigraph_matrices(
                alternate_node, add_upper_bound=True, cost_upper_bound=cost_upper_bound
            )
            T_alt = self.get_projection_transformation(
                node=alternate_node,
                A=A_alt,
                include_cost_epigraph=True,
            )
            P_alt = Polyhedron(A_alt, b_alt)
            P_alt._vertices = VPolytope(HPolyhedron(A_alt, b_alt)).vertices().T

            fig = P_alt.plot_transformation(
                T_alt,
                fig=fig,
                opacity=0.3,
                color="red",
                name=f"Alternate:{alternate_node.vertex_path}",
                showlegend=True,
            )
        # Setting plot layout
        fig.update_layout(
            title="Containment",
        )
        fig.show()