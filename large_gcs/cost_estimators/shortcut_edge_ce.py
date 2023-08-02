from typing import List

from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.graph.graph import Edge, Graph, ShortestPathSolution
from large_gcs.utils.hydra_utils import get_function_from_string


class ShortcutEdgeCE(CostEstimator):
    def __init__(self, graph: Graph, shortcut_edge_cost_factory=None):
        # To allow function string path to be passed in from hydra config
        if type(shortcut_edge_cost_factory) == str:
            shortcut_edge_cost_factory = get_function_from_string(
                shortcut_edge_cost_factory
            )

        if (
            shortcut_edge_cost_factory is None
            and graph._default_costs_constraints.edge_costs is None
        ):
            raise ValueError(
                "If no shortcut_edge_cost_factory is specified, edge costs must be specified in the graph's default costs constraints."
            )
        self._graph = graph
        self._shortcut_edge_cost_factory = shortcut_edge_cost_factory

    def estimate_cost(
        self,
        subgraph: Graph,
        edge: Edge,
        active_edges: List[Edge] = None,
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
            sol = subgraph.solve_shortest_path(
                use_convex_relaxation=use_convex_relaxation
            )

        self._alg_metrics.update_after_gcs_solve(sol.time)

        # Clean up
        subgraph.remove_vertex(neighbor)

        return sol

    @property
    def finger_print(self) -> str:
        return f"ShortcutEdgeCE-{self._shortcut_edge_cost_factory}"
