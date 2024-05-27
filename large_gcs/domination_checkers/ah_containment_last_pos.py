import logging

import numpy as np

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


class ReachesNewLastPosContainment(AHContainmentLastPos):
    @property
    def include_cost_epigraph(self):
        return False


class ReachesCheaperLastPosContainment(AHContainmentLastPos):
    @property
    def include_cost_epigraph(self):
        return True
