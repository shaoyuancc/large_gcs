import logging
from typing import List

from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.graph import Edge, Graph, ShortestPathSolution
from large_gcs.utils.hydra_utils import get_function_from_string

logger = logging.getLogger(__name__)


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

    def estimate_cost_on_graph(
        self,
        graph: Graph,
        edge: Edge,
        active_edges: List[Edge] = None,
        solve_convex_restriction: bool = False,
        use_convex_relaxation: bool = False,
    ) -> ShortestPathSolution:
        neighbor = edge.v

        # Check if this neighbor is the target to see if shortcut edge is required
        # Actually you can't do this because this will make the algorithm assume that neighbor is unreachable, in case you can't get from the neighbor to the target
        add_shortcut_edge = neighbor != self._graph.target_name
        if add_shortcut_edge:
            # Add an edge from the neighbor to the target
            direct_edge_costs = None
            if self._shortcut_edge_cost_factory:
                if isinstance(graph, ContactGraph):
                    # Only ContactSet and ContactPointSet have the vars attribute
                    # convex_sets in general do not.
                    direct_edge_costs = self._shortcut_edge_cost_factory(
                        self._graph.vertices[neighbor].convex_set.vars,
                        self._graph.vertices[self._graph.target_name].convex_set.vars,
                    )
                else:
                    direct_edge_costs = self._shortcut_edge_cost_factory(
                        self._graph.vertices[self._graph.target_name].convex_set.dim,
                    )
            edge_to_target = Edge(
                u=neighbor,
                v=self._graph.target_name,
                key_suffix="shortcut",
                costs=direct_edge_costs,
            )
            graph.add_edge(edge_to_target)
            conv_res_active_edges = active_edges + [edge.key, edge_to_target.key]
        else:
            conv_res_active_edges = active_edges + [edge.key]

        if solve_convex_restriction:
            logger.debug(f"conv_res_active_edges: {conv_res_active_edges}")
            sol = graph.solve_convex_restriction(conv_res_active_edges)
        else:
            sol = graph.solve_shortest_path(use_convex_relaxation=use_convex_relaxation)

        self._alg_metrics.update_after_gcs_solve(sol.time)

        # Clean up
        if add_shortcut_edge:
            logger.debug(f"Removing edge {edge_to_target.key}")
            graph.remove_edge(edge_to_target.key)

        return sol

    def estimate_cost(
        self,
        subgraph: Graph,
        edge: Edge,
        active_edges: List[Edge] = None,
        solve_convex_restriction: bool = False,
        use_convex_relaxation: bool = False,
    ) -> ShortestPathSolution:
        """
        active_edges does not include the edge argument.
        Right now this function is unideally coupled because it returns a shortest path solution instead of just the cost.
        """

        neighbor = edge.v

        # Check if this neighbor is the target to see if shortcut edge is required
        if neighbor != self._graph.target_name:
            # Add neighbor and edge temporarily to the visited subgraph
            subgraph.add_vertex(self._graph.vertices[neighbor], neighbor)
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
            subgraph.add_edge(edge_to_target)
            conv_res_active_edges = active_edges + [edge.key, edge_to_target.key]
        else:
            conv_res_active_edges = active_edges + [edge.key]
        subgraph.add_edge(edge)

        if solve_convex_restriction:
            # logger.debug(f"active edges: {subgraph.edges.keys()}")
            sol = subgraph.solve_convex_restriction(conv_res_active_edges)
        else:
            sol = subgraph.solve_shortest_path(
                use_convex_relaxation=use_convex_relaxation
            )

        self._alg_metrics.update_after_gcs_solve(sol.time)

        # Clean up
        if neighbor != self._graph.target_name:
            subgraph.remove_vertex(neighbor)
        else:
            subgraph.remove_edge(edge.key)

        return sol

    @property
    def finger_print(self) -> str:
        return f"ShortcutEdgeCE-{self._shortcut_edge_cost_factory}"
