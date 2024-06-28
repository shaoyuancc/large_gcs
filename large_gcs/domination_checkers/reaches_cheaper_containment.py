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
    @property
    def include_cost_epigraph(self):
        return True

    def plot_containment(
        self,
        candidate_node: SearchNode,
        alternate_nodes: List[SearchNode],
        cost_upper_bound: float = 20,
    ):
        prog = self.get_path_mathematical_program(
            candidate_node, add_upper_bound=True, cost_upper_bound=cost_upper_bound
        )
        h_poly = HPolyhedron(prog)
        P_n = Polyhedron(h_poly.A(), h_poly.b(), should_compute_vertices=False)
        T_n = self.get_H_transformation(
            node=candidate_node,
            total_dims=h_poly.ambient_dimension(),
        )
        P_n._vertices = VPolytope(h_poly).vertices().T
        fig = P_n.plot_transformation(
            T_n,
            color="blue",
            name=f"Candidate:{candidate_node.vertex_path}",
        )
        for alternate_node in alternate_nodes:
            prog_alt = self.get_path_mathematical_program(
                alternate_node, add_upper_bound=True, cost_upper_bound=cost_upper_bound
            )
            h_poly_alt = HPolyhedron(prog_alt)
            T_alt = self.get_H_transformation(
                node=alternate_node,
                total_dims=h_poly_alt.ambient_dimension(),
            )
            P_alt = Polyhedron(
                h_poly_alt.A(), h_poly_alt.b(), should_compute_vertices=False
            )
            P_alt._vertices = VPolytope(h_poly_alt).vertices().T

            fig = P_alt.plot_transformation(
                T_alt,
                fig=fig,
                color="red",
                name=f"Alternate:{alternate_node.vertex_path}",
            )
        fig.show()
