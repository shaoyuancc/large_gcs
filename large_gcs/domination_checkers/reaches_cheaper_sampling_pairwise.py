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


class ReachesCheaperSamplingPairwise(SamplingDominationChecker):
    def is_dominated(
        self, candidate_node: SearchNode, alternate_nodes: List[SearchNode]
    ) -> bool:
        """Checks to see if there is a single alternate path that reaches every
        sample cheaper than the candidate path.

        If so, the candidate path is dominated. This is explictly not
        checking the candidate against the union of the alternate paths.
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
        candidate_costs = np.full(projected_samples.shape[0], np.inf)

        # Add all sample vertices to graph and collate costs via candidate path
        for idx, proj_sample in enumerate(projected_samples):
            # Create a new vertex for the sample and add it to the graph
            sample_vertex_name = sample_names[idx]

            self._graph.add_vertex(
                vertex=Vertex(convex_set=Point(proj_sample)), name=sample_vertex_name
            )

            sol = self._solve_conv_res_to_sample(candidate_node, sample_vertex_name)

            if not sol.is_success:
                logger.error(
                    f"Candidate path was not feasible to reach sample {idx}"
                    f"\nnum samples: {len(self._set_samples[candidate_node.vertex_name].samples)}"
                    f"\nsample: {proj_sample}"
                    f"\nproj_sample: {proj_sample}"
                    f"\nactive edges: {candidate_node.edge_path}"
                    f"\nvertex_path: {candidate_node.vertex_path}"
                    f"\n Skipping to next sample"
                )
                # assert sol.is_success, "Candidate path should be feasible"
                continue
            candidate_costs[idx] = sol.cost

        # Check if any alternate path reaches every sample cheaper than candidate path
        for alt_n in alternate_nodes:
            alt_costs = np.full_like(candidate_costs, np.inf)
            for idx, sample_vertex_name in enumerate(sample_names):
                if candidate_costs[idx] == np.inf:
                    continue

                sol = self._solve_conv_res_to_sample(alt_n, sample_vertex_name)
                if sol.is_success:
                    alt_costs[idx] = sol.cost
            if np.all(alt_costs <= candidate_costs):
                is_dominated = True
                break

        # Clean up
        for sample_vertex_name in sample_names:
            self._graph.remove_vertex(sample_vertex_name)

        self._graph.set_target(self._target)
        return is_dominated
