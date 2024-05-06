import logging
from typing import List

from large_gcs.algorithms.search_algorithm import SearchNode
from large_gcs.domination_checkers.sampling_domination_checker import (
    SamplingDominationChecker,
    SetSamples,
)
from large_gcs.geometry.point import Point
from large_gcs.graph.graph import Edge, Vertex

logger = logging.getLogger(__name__)


class ReachesNewSampling(SamplingDominationChecker):
    """
    Checks samples to see if this path reaches new previously unreached samples.
    Assumes that this path is feasible.
    """

    def is_dominated(
        self, candidate_node: SearchNode, alternate_nodes: List[SearchNode]
    ) -> bool:
        """
        A candidate path is dominated if it does not reach any new regions in the set.
        """
        self._maybe_add_set_samples(candidate_node.vertex_name)

        reached_new = False
        projected_samples = set()
        self._set_samples[candidate_node.vertex_name].init_graph_for_projection(
            self._graph, candidate_node, self._alg_metrics
        )
        for idx, sample in enumerate(
            self._set_samples[candidate_node.vertex_name].samples
        ):
            sample = self._set_samples[candidate_node.vertex_name].project_single_gcs(
                self._graph, candidate_node, sample
            )

            if sample is None:
                logger.warning(
                    f"Failed to project sample {idx} for vertex {candidate_node.vertex_name}"
                )
                continue
            else:
                if tuple(sample) in projected_samples:
                    logger.debug(f"projected sample {idx} same as a previous sample")
                    continue
                projected_samples.add(tuple(sample))
            # Create a new vertex for the sample and add it to the graph
            sample_vertex_name = f"{candidate_node.vertex_name}_sample_{idx}"
            self._graph.add_vertex(
                vertex=Vertex(convex_set=Point(sample)), name=sample_vertex_name
            )
            go_to_next_sample = False
            for alt_n in alternate_nodes:
                sol = self._solve_conv_res_to_sample(alt_n, sample_vertex_name)
                if sol.is_success:
                    # Clean up current sample
                    self._graph.remove_vertex(sample_vertex_name)
                    # Move on to the next sample, don't need to check other paths
                    # logger.debug(f"Sample {idx} reached by path {alt_n.vertex_path}")
                    go_to_next_sample = True
                    break
            if go_to_next_sample:
                continue
            # If no paths can reach the sample, do not need to check more samples
            reached_new = True
            # Clean up
            if sample_vertex_name in self._graph.vertices:
                self._graph.remove_vertex(sample_vertex_name)

            logger.debug(f"Sample {idx} not reached by any previous path.")
            break
        self._graph.set_target(self._target)
        return not reached_new
