import logging
from typing import Optional

from large_gcs.algorithms.search_algorithm import SearchNode
from large_gcs.contact.contact_set import ContactPointSet, ContactSet
from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.graph.graph import Edge, Graph, ShortestPathSolution
from large_gcs.utils.hydra_utils import get_function_from_string

logger = logging.getLogger(__name__)


class ShortcutEdgeCE(CostEstimator):
    def __init__(
        self,
        graph: Graph,
        shortcut_edge_cost_factory=None,
        add_const_cost: bool = False,
    ):
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
        self._add_const_cost = add_const_cost

    def estimate_cost(
        self,
        graph: Graph,
        successor: str,
        node: SearchNode,
        heuristic_inflation_factor: float,
        solve_convex_restriction: bool = False,
        use_convex_relaxation: bool = False,
        override_skip_post_solve: Optional[bool] = None,
    ) -> ShortestPathSolution:

        # Check if this neighbor is the target to see if shortcut edge is required
        add_shortcut_edge = successor != self._graph.target_name
        edge_to_successor = Edge.key_from_uv(node.vertex_name, successor)
        if add_shortcut_edge:
            # Add an edge from the neighbor to the target
            direct_edge_costs = None
            if self._shortcut_edge_cost_factory:
                if isinstance(
                    graph.vertices[successor].convex_set, ContactSet
                ) or isinstance(graph.vertices[successor], ContactPointSet):
                    # Only ContactSet and ContactPointSet have the vars attribute
                    # convex_sets in general do not.
                    direct_edge_costs = self._shortcut_edge_cost_factory(
                        u_vars=self._graph.vertices[successor].convex_set.vars,
                        v_vars=self._graph.vertices[
                            self._graph.target_name
                        ].convex_set.vars,
                        heuristic_inflation_factor=heuristic_inflation_factor,
                        add_const_cost=self._add_const_cost,
                    )

                else:
                    direct_edge_costs = self._shortcut_edge_cost_factory(
                        self._graph.vertices[successor].convex_set.dim,
                        heuristic_inflation_factor=heuristic_inflation_factor,
                        add_const_cost=self._add_const_cost,
                    )

            edge_to_target = Edge(
                u=successor,
                v=self._graph.target_name,
                key_suffix="shortcut",
                costs=direct_edge_costs,
            )
            graph.add_edge(edge_to_target)
            conv_res_active_edges = node.edge_path + [
                edge_to_successor,
                edge_to_target.key,
            ]
        else:
            conv_res_active_edges = node.edge_path + [edge_to_successor]

        if solve_convex_restriction:
            skip_post_solve = (
                add_shortcut_edge
                if override_skip_post_solve is None
                else override_skip_post_solve
            )
            # If used shortcut edge, do not parse the full result since we won't use the solution.
            sol = graph.solve_convex_restriction(
                conv_res_active_edges, skip_post_solve=skip_post_solve
            )
        else:
            sol = graph.solve_shortest_path(use_convex_relaxation=use_convex_relaxation)

        self._alg_metrics.update_after_gcs_solve(sol.time)

        # Clean up
        if add_shortcut_edge:
            logger.debug(f"Removing edge {edge_to_target.key}")
            graph.remove_edge(edge_to_target.key)

        return sol

    @property
    def finger_print(self) -> str:
        return f"ShortcutEdgeCE-{self._shortcut_edge_cost_factory.__name__}"
