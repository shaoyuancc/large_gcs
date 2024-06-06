import logging

import numpy as np
from scipy.linalg import block_diag

from large_gcs.algorithms.search_algorithm import SearchNode
from large_gcs.contact.contact_set import ContactSet
from large_gcs.domination_checkers.ah_containment_domination_checker import (
    AHContainmentDominationChecker,
)
from large_gcs.geometry.geometry_utils import create_selection_matrix

logger = logging.getLogger(__name__)


class AHContainmentLastPos(AHContainmentDominationChecker):
    def get_H_transformation(
        self,
        node: SearchNode,
        total_dims: int,
    ):
        """Get the transformation matrix that will project the polyhedron that
        defines the whole path down to just the dimensions of the selected
        vertex.

        Can either include the epigraph (include the cost) or just the
        dimensions of the vertex.
        Note: Cost epigraph variable assumed to be the last decision variable in x.
        """
        # logger.debug(f"get_H_transformation")
        # First, collect all the decision variables
        v_dims = [
            self._graph.vertices[name].convex_set.dim for name in node.vertex_path
        ]
        current_index = 0
        # Collect the indices of the decision variables for each vertex
        x = []
        for dim in v_dims:
            x.append(list(range(current_index, current_index + dim)))
            current_index += dim
        terminal_set: ContactSet = self._graph.vertices[node.vertex_name].convex_set
        selected_indices = list(terminal_set.vars.last_pos_from_all(x[-1]))
        if self.include_cost_epigraph:
            # Assumes the cost variable is the last variable
            selected_indices.append(total_dims - 1)
        return create_selection_matrix(selected_indices, total_dims)

    def get_nullspace_H_transformation(
        self,
        node: SearchNode,
        full_dim: int,
        ns_dim: int,
    ):
        # logger.debug(f"get_nullspace_H_transformation")
        """Get the transformation matrix that will project the polyhedron that
        defines the whole path down to just the dimensions of the last vertex's
        nullspace.

        Can either include the epigraph (include the cost) or just the
        dimensions of the vertex.
        Note: Cost epigraph variable assumed to be the last decision variable in x.
        """
        S = self.get_H_transformation(node, full_dim)

        Vs = block_diag(
            *[
                self._graph.vertices[name].convex_set.nullspace_set._V
                for name in node.vertex_path
            ]
        )
        x_0s = np.concatenate(
            [
                self._graph.vertices[name].convex_set.nullspace_set._x_0
                for name in node.vertex_path
            ]
        )

        T = S @ Vs
        t = S @ x_0s
        return T, t


class ReachesNewLastPosContainment(AHContainmentLastPos):
    @property
    def include_cost_epigraph(self):
        return False


class ReachesCheaperLastPosContainment(AHContainmentLastPos):
    @property
    def include_cost_epigraph(self):
        return True
