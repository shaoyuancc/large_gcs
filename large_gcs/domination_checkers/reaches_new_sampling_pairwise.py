import logging
from typing import List

import numpy as np

from large_gcs.algorithms.search_algorithm import SearchNode
from large_gcs.domination_checkers.sampling_domination_checker import (
    SamplingDominationChecker,
    SetSamples,
)
from large_gcs.geometry.point import Point
from large_gcs.graph.graph import Edge, Vertex

logger = logging.getLogger(__name__)


class ReachesNewSamplingPairwise(SamplingDominationChecker):
    def is_dominated(
        self, candidate_node: SearchNode, alternate_nodes: List[SearchNode]
    ) -> bool:
        """
        Checks to see if there is a single alternate path that reaches every sample that the candidate path reaches.
        If so, the candidate path is dominated.
        This is explictly not checking the candidate against the union of the alternate paths.
        """
        self._maybe_add_set_samples(candidate_node.vertex_name)

        is_dominated = False
        # First project all the points
        projected_samples = self._set_samples[
            candidate_node.vertex_name
        ].project_all_gcs(self._graph, candidate_node, self._alg_metrics)
        sample_names = [
            f"{candidate_node.vertex_name}_sample_{idx}"
            for idx in range(len(projected_samples))
        ]

        # Add all sample vertices to graph
        for idx, proj_sample in enumerate(projected_samples):
            # Create a new vertex for the sample and add it to the graph
            sample_vertex_name = sample_names[idx]

            self._graph.add_vertex(
                vertex=Vertex(convex_set=Point(proj_sample)), name=sample_vertex_name
            )

        # Assumes that the candidate path can reach all projected samples
        # But currently there is some bug that causes solve_convex_restriction to the sample
        # to fail for some samples for the candidate path.

        # Check if any alternate path reaches every sample
        for alt_n in alternate_nodes:
            alt_reaches = np.full(len(sample_names), False)
            for idx, sample_vertex_name in enumerate(sample_names):
                sol = self._solve_conv_res_to_sample(alt_n, sample_vertex_name)
                alt_reaches[idx] = sol.is_success
            if np.all(alt_reaches):
                is_dominated = True
                break

        # Clean up
        for sample_vertex_name in sample_names:
            self._graph.remove_vertex(sample_vertex_name)

        self._graph.set_target(self._target)
        return is_dominated
