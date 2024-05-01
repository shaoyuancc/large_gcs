from typing import List
import logging
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
        if candidate_node.vertex_name not in self._set_samples:
            logger.debug(f"Adding samples for {candidate_node.vertex_name}")
            self._set_samples[candidate_node.vertex_name] = SetSamples.from_vertex(
                candidate_node.vertex_name,
                self._graph.vertices[candidate_node.vertex_name],
                self._num_samples_per_vertex,
            )

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
                logger.warn(
                    f"Failed to project sample {idx} for vertex {candidate_node.vertex_name}"
                )
                continue
            else:
                if tuple(sample) in projected_samples:
                    continue
                projected_samples.add(tuple(sample))
            # Create a new vertex for the sample and add it to the graph
            sample_vertex_name = f"{candidate_node.vertex_name}_sample_{idx}"
            self._graph.add_vertex(
                vertex=Vertex(convex_set=Point(sample)), name=sample_vertex_name
            )
            go_to_next_sample = False
            for alt_n in alternate_nodes:
                # Add edge between the sample and the second last vertex in the path
                e = self._graph.edges[alt_n.edge_path[-1]]
                edge_to_sample = Edge(
                    u=e.u,
                    v=sample_vertex_name,
                    costs=e.costs,
                    constraints=e.constraints,
                )
                self._graph.add_edge(edge_to_sample)
                # Check whether sample can be reached via the path
                self._graph.set_target(sample_vertex_name)
                active_edges = alt_n.edge_path.copy()
                active_edges[-1] = edge_to_sample.key

                sol = self._graph.solve_convex_restriction(
                    active_edges, skip_post_solve=True
                )
                self._alg_metrics.update_after_gcs_solve(sol.time)
                if sol.is_success:
                    # Clean up current sample
                    self._graph.remove_vertex(sample_vertex_name)
                    # Move on to the next sample, don't need to check other paths
                    # logger.debug(f"Sample {idx} reached by path {alt_n.vertex_path}")
                    go_to_next_sample = True
                    break
                else:
                    # Clean up edge, but leave the sample vertex
                    self._graph.remove_edge(edge_to_sample.key)
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
