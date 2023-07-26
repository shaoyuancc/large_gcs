from itertools import combinations, product

from tqdm import tqdm

from large_gcs.contact.contact_pair_mode import generate_no_contact_pair_modes
from large_gcs.contact.contact_set import ContactSet
from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.factored_collision_free_graph import FactoredCollisionFreeGraph
from large_gcs.graph.graph import Edge, Graph, ShortestPathSolution


class FactoredCollisionFreeCE(CostEstimator):
    def __init__(self, graph: ContactGraph):
        self._graph = graph

        self._collision_free_graphs = [
            FactoredCollisionFreeGraph(
                body,
                self._graph.obstacles,
                self._graph.target_pos[i],
                self._graph.workspace,
            )
            for i, body in enumerate(self._graph.objects + self._graph.robots)
        ]

    def estimate_cost(
        self,
        subgraph: Graph,
        edge: Edge,
        solve_convex_restriction: bool = False,
        use_convex_relaxation: bool = False,
    ) -> ShortestPathSolution:
        """Right now this function is unideally coupled because it returns a shortest path solution instead of just the cost."""

        neighbor = edge.v
        # Add neighbor and edge temporarily to the visited subgraph
        subgraph.add_vertex(self._graph.vertices[neighbor], neighbor)
        # Check if this neighbor actually has an edge to the target
        # If so, add that edge instead of the shortcut
        if (neighbor, self._graph.target_name) in self._graph.edges:
            edge_to_target = self._graph.edges[(neighbor, self._graph.target_name)]
        else:
            # Add an edge from the neighbor to the target
            direct_edge_costs = None
            if self._shortcut_edge_cost_factory:
                # Note for now this only works with ContactSet and ContactPointSet because
                # they have the vars attribute, and convex_sets in general do not.
                direct_edge_costs = self._shortcut_edge_cost_factory(
                    self._graph.vertices[neighbor].convex_set.vars,
                    self._graph.vertices[self._graph.target_name].convex_set.vars,
                )
            edge_to_target = Edge(
                neighbor, self._graph.target_name, costs=direct_edge_costs
            )
        subgraph.add_edge(edge)
        subgraph.add_edge(edge_to_target)

        if solve_convex_restriction:
            sol = subgraph.solve_convex_restriction(subgraph.edges.values())
        else:
            sol = subgraph.solve(use_convex_relaxation=use_convex_relaxation)

        self._alg_metrics.update_after_gcs_solve(sol.time)

        # Clean up
        subgraph.remove_vertex(neighbor)

        return sol
