import logging
from typing import List

from large_gcs.algorithms.search_algorithm import SearchNode
from large_gcs.contact.contact_set import ContactSet
from large_gcs.domination_checkers.sampling_domination_checker import (
    SamplingDominationChecker,
    SetSamples,
)
from large_gcs.geometry.point import Point
from large_gcs.graph.contact_cost_constraint_factory import vertex_constraint_last_pos
from large_gcs.graph.graph import Edge, Graph, Vertex

logger = logging.getLogger(__name__)


class ReachesNewLastPosSampling(SamplingDominationChecker):
    """
    Checks samples to see if this path reaches new previously unreached samples.
    Assumes that this path is feasible.
    """

    def is_dominated(
        self, candidate_node: SearchNode, alternate_nodes: List[SearchNode]
    ) -> bool:
        """
        A candidate path is dominated if it does not reach any new last position knot points in the set.
        """
        self._maybe_add_set_samples(candidate_node.vertex_name)

        contact_set: ContactSet = self._graph.vertices[
            candidate_node.vertex_name
        ].convex_set
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
                last_pos_sample = contact_set.vars.last_pos_from_all(sample)
                if tuple(last_pos_sample) in projected_samples:
                    logger.debug(f"projected sample {idx} same as a previous sample")
                    continue
                projected_samples.add(tuple(last_pos_sample))
            # Create a new vertex for the sample and add it to the graph
            sample_vertex_name = f"{candidate_node.vertex_name}_sample_{idx}"
            self._graph.add_vertex(
                vertex=Vertex(
                    convex_set=contact_set,
                    constraints=[vertex_constraint_last_pos(contact_set.vars, sample)],
                ),
                name=sample_vertex_name,
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
