import ast
import logging
import re

import numpy as np
from tqdm import tqdm

from large_gcs.contact.rigid_body import MobilityType
from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.factored_collision_free_graph import FactoredCollisionFreeGraph
from large_gcs.graph.graph import Edge, Graph, ShortestPathSolution

logger = logging.getLogger(__name__)


class FactoredCollisionFreeCE(CostEstimator):
    def __init__(
        self,
        graph: ContactGraph,
        add_transition_cost: bool = True,
        obj_multiplier: float = 1.0,
    ):
        self._graph = graph
        self._add_transition_cost = add_transition_cost
        self._obj_multiplier = obj_multiplier
        logger.info(
            f"creating {self._graph.n_objects + self._graph.n_robots} collision free graphs..."
        )
        self._collision_free_graphs = [
            FactoredCollisionFreeGraph(
                body,
                self._graph.obstacles,
                self._graph.target_pos[i],
                self._graph.workspace,
            )
            for i, body in tqdm(enumerate(self._graph.objects + self._graph.robots))
        ]

        # Look up table of cfree vertex name to cfree cost
        self._cfree_cost = {}
        self._cfree_init_pos = {}

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
        subgraph.add_edge(edge)
        # Check if this neighbor actually has an edge to the target
        # If so, add that edge instead of calculating the collision free cost
        neighbor_has_edge_to_target = (
            neighbor,
            self._graph.target_name,
        ) in self._graph.edges
        if neighbor_has_edge_to_target:
            edge_to_target = self._graph.edges[(neighbor, self._graph.target_name)]
            subgraph.add_edge(edge_to_target)
            subgraph.set_target(self._graph.target_name)
        else:
            # set the neighbor as the target
            subgraph.set_target(neighbor)

        if solve_convex_restriction:
            sol = subgraph.solve_convex_restriction(subgraph.edges.values())
        else:
            sol = subgraph.solve(use_convex_relaxation=use_convex_relaxation)
        self._alg_metrics.update_after_gcs_solve(sol.time)
        # Clean up
        subgraph.remove_vertex(neighbor)

        if sol.is_success and not neighbor_has_edge_to_target:
            # Position within neighbor that the first solve ends at
            x_vars = self._graph.vertices[neighbor].convex_set.vars
            x_pos = x_vars.pos_from_all(sol.ambient_path[-1])

            # Calculate or look up the collision free cost for each body
            cfree_cost = 0
            for i, cfree_vertex_name in enumerate(self.split_vertex_names(neighbor)):
                body_pos_end = x_pos[i].T[-1].flatten()
                g = self._collision_free_graphs[i]
                if cfree_vertex_name in self._cfree_cost:
                    logger.debug(
                        f"Using cached cfree cost for {cfree_vertex_name}, cost: {self._cfree_cost[cfree_vertex_name]}"
                    )
                    cfree_cost += self._cfree_cost[cfree_vertex_name]
                    cfree_init_pos = self._cfree_init_pos[cfree_vertex_name]
                else:
                    g.set_source(cfree_vertex_name)
                    cfree_sol = g.solve(use_convex_relaxation=use_convex_relaxation)
                    if cfree_sol.is_success:
                        new_cfree_cost = (
                            self._obj_multiplier * cfree_sol.cost
                            if g.movable_body.mobility_type == MobilityType.UNACTUATED
                            else cfree_sol.cost
                        )
                        cfree_cost += new_cfree_cost
                        self._cfree_cost[cfree_vertex_name] = new_cfree_cost
                        x_cfree_vars = g.vertices[cfree_vertex_name].convex_set.vars
                        cfree_init_pos = (
                            x_cfree_vars.pos_from_all(cfree_sol.ambient_path[0])
                            .T[0]
                            .flatten()
                        )
                        self._cfree_init_pos[cfree_vertex_name] = cfree_init_pos
                        logger.debug(
                            f"Calculated cfree cost for {cfree_vertex_name}, cost: {cfree_sol.cost}"
                        )
                    else:
                        cfree_cost += float("inf")
                        self._cfree_cost[cfree_vertex_name] = float("inf")
                        self._cfree_init_pos[cfree_vertex_name] = body_pos_end
                        cfree_init_pos = body_pos_end
                        logger.warn(
                            f"Could not find collision free path for {cfree_vertex_name}"
                        )

                    self._cfree_cost[cfree_vertex_name] = cfree_cost
                    self._alg_metrics.update_after_gcs_solve(cfree_sol.time)

                if self._add_transition_cost:
                    transition_cost = np.linalg.norm(body_pos_end - cfree_init_pos)
                    cfree_cost += (
                        self._obj_multiplier * transition_cost
                        if g.movable_body.mobility_type == MobilityType.UNACTUATED
                        else transition_cost
                    )
            logger.debug(
                f"explored {neighbor} cost to come: {sol.cost}, cfree cost: {cfree_cost}, total cost: {sol.cost + cfree_cost}"
            )
            # Add heuristic cost to cost to come
            sol.cost += cfree_cost

        return sol

    @staticmethod
    def _find_obj_rob_numbers(s: str):
        match = re.search(r"(obj|rob)(\d+)", s)
        if match:
            return match.group(
                2
            )  # group 2 is the second capture group, which contains the digits
        else:
            return None

    @staticmethod
    def split_vertex_names(vertex_name: str):
        # Convert string representation of tuple to actual tuple
        tuple_vertex = ast.literal_eval(vertex_name)

        # Initialize dictionaries to store modes for each obj and rob
        obj_modes = {}
        rob_modes = {}

        for mode in tuple_vertex:
            # Check if mode contains both obj and rob
            if "obj" in mode and "rob" in mode:
                continue

            # Extract the entity number from the mode string
            entity_num = FactoredCollisionFreeCE._find_obj_rob_numbers(mode)
            # Add mode to appropriate dictionary
            if "obj" in mode:
                if entity_num not in obj_modes:
                    obj_modes[entity_num] = []
                obj_modes[entity_num].append(mode)
            elif "rob" in mode:
                if entity_num not in rob_modes:
                    rob_modes[entity_num] = []
                rob_modes[entity_num].append(mode)

        # Combine mode dictionaries into a list of tuples
        vertex_res = []
        for entity_num, modes in obj_modes.items():
            vertex_res.append(str(tuple(modes)))
        for entity_num, modes in rob_modes.items():
            vertex_res.append(str(tuple(modes)))
        return vertex_res

    @property
    def finger_print(self) -> str:
        return (
            f"FactoredCollisionFreeCE-add_transition_cost-{self._add_transition_cost}"
        )
