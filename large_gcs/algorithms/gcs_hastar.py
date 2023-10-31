import heapq as heap
import logging
import time
from copy import deepcopy
from typing import List, Optional

import numpy as np

from large_gcs.abstraction_models.abstraction_model import AbstractionModel
from large_gcs.abstraction_models.gcshastar_node import (
    ContextNode,
    GCSHANode,
    StatementNode,
)
from large_gcs.algorithms.search_algorithm import (
    AlgMetrics,
    AlgVisParams,
    ReexploreLevel,
    SearchAlgorithm,
)
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.factored_collision_free_graph import FactoredCollisionFreeGraph
from large_gcs.graph.graph import Graph, ShortestPathSolution
from large_gcs.graph.incremental_contact_graph import IncrementalContactGraph
from large_gcs.graph_generators.contact_graph_generator import (
    ContactGraphGeneratorParams,
)

logger = logging.getLogger(__name__)


class GcsHAstarMetrics(AlgMetrics):
    def initialize(self, n_levels: int):
        empty_dict = {
            i: {
                StatementNode.__name__: 0,
                ContextNode.__name__: 0,
            }
            for i in range(n_levels)
        }
        self.n_vertices_expanded = deepcopy(empty_dict)
        self.n_vertices_reexpanded = deepcopy(empty_dict)
        self.n_vertices_visited = deepcopy(empty_dict)
        self.n_vertices_revisited = deepcopy(empty_dict)


class GcsHAstar(SearchAlgorithm):
    def __init__(
        self,
        abs_model: AbstractionModel,
        reexplore_levels: Optional[List[ReexploreLevel]] = None,
        vis_params: AlgVisParams = AlgVisParams(),
    ):
        self._graphs: List[ContactGraph] = abs_model.graphs
        self._abs_fns = abs_model.abs_fns
        if reexplore_levels is None:
            self._reexplore_levels = [ReexploreLevel.NONE] * len(self._graphs)
        else:
            self._reexplore_levels = [
                ReexploreLevel[reexplore_level]
                if type(reexplore_level) == str
                else reexplore_level
                for reexplore_level in reexplore_levels
            ]
        self._vis_params = vis_params

        self._S: dict[str, GCSHANode] = {}  # Expanded/Closed set
        self._Q: list[GCSHANode] = []  # Priority queue
        max_abs_level = len(self._graphs)
        self._alg_metrics = GcsHAstarMetrics()
        self._alg_metrics.initialize(n_levels=max_abs_level + 1)
        start_node = StatementNode(
            priority=0, abs_level=max_abs_level, vertex_name="START", path=[], weight=0
        )
        heap.heappush(self._Q, start_node)

        # Compile targets at each level of abstraction
        self._targets = {
            i: self._graphs[i].target_name for i in range(len(self._graphs))
        }
        self._targets[max_abs_level] = "START"
        self._sources = {
            i: self._graphs[i].source_name for i in range(len(self._graphs))
        }

        self._iteration = 0

    def run(self):
        self._start_time = time.time()

        sol = None
        while sol == None and len(self._Q) > 0:
            sol = self._run_iteration()
            self._alg_metrics.time_wall_clock = time.time() - self._start_time
        if sol is None:
            logger.warn(
                f"Gcs HA* Convex Restriction failed to find a path to the target."
            )
            return

        g = self._graphs[0]
        g._post_solve(sol)
        logger.info(
            f"Gcs HA* Convex Restriction complete! \ncost: {sol.cost}, time: {sol.time}\nvertex path: {np.array(sol.vertex_path)}\n{self.alg_metrics}"
        )

    def _run_iteration(self):
        self._iteration += 1
        # print(f"\niteration: {self._iteration}")
        # print(f"S {[(key, val.weight) for (key, val) in self._S.items()]}")
        # # Make a copy of the priority queue.
        # pq_copy = copy(self._Q)
        # # Pop the top 10 items from the priority queue copy.
        # bottom_10 = []
        # for _ in range(min(10, len(pq_copy))):
        #     n = heap.heappop(pq_copy)
        #     bottom_10.append((n.id, n.priority))
        # print(f"Lowest 10 in Q: {bottom_10}")

        n: GCSHANode = heap.heappop(self._Q)
        if not self._should_reexpand(n):
            return

        if n.id in self._S:
            self._alg_metrics.n_vertices_reexpanded[n.abs_level][type(n).__name__] += 1
        else:
            self._alg_metrics.n_vertices_expanded[n.abs_level][type(n).__name__] += 1

        logger.info(f"\n{self.alg_metrics}\nexpanding {n.id}")
        self._S[n.id] = n
        self.log_metrics_to_wandb(n.priority)

        if isinstance(n, StatementNode):
            # Check if BASE rule applies (the goal of a level is reached)
            if n.vertex_name == self._targets[n.abs_level]:
                logger.info(f"Goal reached at level {n.abs_level}")
                self._execute_base_rule(n)
                if n.abs_level == 0:
                    return n.sol
                else:
                    return None

            # Generate neighbors (vertices and edges) that you are about to explore
            g = self._graphs[n.abs_level]
            g.generate_neighbors(n.vertex_name)
            # Get UP rules (edges at the same level)
            edges = g.outgoing_edges(n.vertex_name)
            if self._reexplore_levels[n.abs_level] == ReexploreLevel.NONE:
                for edge in edges:
                    neighbor = StatementNode.from_parent(
                        child_vertex_name=edge.v, parent=n
                    )
                    if neighbor.id not in self._S:  # No revisiting
                        abs_neighbor_nodes = self._abs_fns[n.abs_level](neighbor)
                        # Check if all required contexts of the abstracted neighbor are in S
                        if all(
                            [
                                abs_n.context_id in self._S
                                for abs_n in abs_neighbor_nodes
                            ]
                        ):
                            self._execute_up_rule(neighbor)
                        else:
                            logger.debug(
                                f"required contexts for {neighbor.id} not in S , continuing"
                            )
            else:
                for edge in edges:
                    neighbor_in_path = any(
                        (u == edge.v or v == edge.v) for (u, v) in n.path
                    )
                    if not neighbor_in_path:
                        neighbor = StatementNode.from_parent(
                            child_vertex_name=edge.v, parent=n
                        )
                        abs_neighbor_nodes = self._abs_fns[n.abs_level](neighbor)
                        # Check if all required contexts of the abstracted neighbor are in S
                        if all(
                            [
                                abs_n.context_id in self._S
                                for abs_n in abs_neighbor_nodes
                            ]
                        ):
                            self._execute_up_rule(neighbor)
                        else:
                            logger.debug(
                                f"required contexts for {neighbor.id} not in S , continuing"
                            )

        # CONTEXT and has other statements in path to convert to contexts
        elif isinstance(n, ContextNode):
            if len(n.path) > 0:
                # Get DOWN rules
                # Go back one step along the path that was taken to calculate its context
                self._execute_down_rule(n_antecedent=n)
            else:
                # The context for the source node has been added to S, so we can now add the source node to Q
                self._execute_source_up_rule(n)

    def _execute_base_rule(self, n_antecedent: StatementNode):
        logger.info(f"Executing BASE rule for antecendent {n_antecedent.id}")
        # The antecedent will always be a goal node.

        # Extract path costs
        path_costs = []
        if len(n_antecedent.path) > 0:
            original_g: ContactGraph = self._graphs[n_antecedent.abs_level]
            # WORKAROUND individual costs not being available in solve convex restriction result.
            # Create as new graph with just the path from the source to the goal and solve that as a full problem
            temp_g = Graph()
            temp_g.add_vertex(original_g.source, original_g.source_name)
            for (u, v) in n_antecedent.path:
                temp_g.add_vertex(original_g.vertices[v], v)
                temp_g.add_edge(original_g.edges[(u, v)])
            temp_g.set_source(original_g.source_name)
            temp_g.set_target(original_g.target_name)
            sol = temp_g.solve_shortest_path()
            n_antecedent.sol = sol
            g = temp_g

            for e in n_antecedent.path:
                vertex_cost = g.vertices[e[1]].gcs_vertex.GetSolutionCost(
                    n_antecedent.sol.result
                )
                edge_cost = g.edges[e].gcs_edge.GetSolutionCost(n_antecedent.sol.result)
                path_costs.append(vertex_cost + edge_cost)

        n_next = ContextNode(
            # Priority of the context is the weight of the statement
            priority=n_antecedent.weight,
            # Weight of the context of the goal statement is 0
            weight=0,
            abs_level=n_antecedent.abs_level,
            vertex_name=n_antecedent.vertex_name,
            path=n_antecedent.path,
            path_costs=path_costs,
            sol=n_antecedent.sol,
        )
        self._update_vertex_visit_revisit(n_next)
        heap.heappush(self._Q, n_next)

    def _execute_down_rule(self, n_antecedent: ContextNode):
        logger.info(f"Executing DOWN rule for antecedent {n_antecedent.id}")

        child = ContextNode(
            priority=n_antecedent.sol.cost,
            abs_level=n_antecedent.abs_level,
            vertex_name=n_antecedent.path[-1][0],
            # Weight of the prior context is parent's weight + vertex cost and edge cost (stored in path_costs)
            weight=n_antecedent.weight + n_antecedent.path_costs[-1],
            path=n_antecedent.path[:-1],
            path_costs=n_antecedent.path_costs[:-1],
            sol=n_antecedent.sol,
            parent=n_antecedent,
        )

        self._update_vertex_visit_revisit(child)
        heap.heappush(self._Q, child)

    def _execute_source_up_rule(self, n_antecedent: ContextNode):
        logger.info(f"Executing SOURCE UP rule for antecedent {n_antecedent.id}")

        lower_abs_level = n_antecedent.abs_level - 1

        source_name = self._graphs[lower_abs_level].source_name
        # add the source node of the next level to the queue
        n_source = StatementNode(
            priority=n_antecedent.priority,  # We want this to get popped off immediately after the context above is popped off
            abs_level=lower_abs_level,
            vertex_name=source_name,
            path=[],
            weight=0,
        )
        self._update_vertex_visit_revisit(n_source)
        heap.heappush(self._Q, n_source)

    def _execute_up_rule(self, n_conclusion: GCSHANode):
        logger.debug(f"Executing UP rule for conclusion {n_conclusion.id}")

        self._update_vertex_visit_revisit(n_conclusion)

        g: ContactGraph = self._graphs[n_conclusion.abs_level]
        abs_fn = self._abs_fns[n_conclusion.abs_level]

        # Solve convex restriction on the path from the source to the conclusion to get the weight
        g.set_target(n_conclusion.vertex_name)
        sol = g.solve_convex_restriction(n_conclusion.path)
        self._alg_metrics.update_after_gcs_solve(sol.time)
        g.set_target(self._targets[n_conclusion.abs_level])
        if not sol.is_success:
            logger.debug(
                f"edge {n_conclusion.parent.id} -> {n_conclusion.id} not actually feasible"
            )
            # Conclusion invalid, do nothing, don't add to Q
            return
        n_conclusion.sol = sol
        n_conclusion.weight = sol.cost
        abs_nodes = abs_fn(n_conclusion)
        n_conclusion.priority = n_conclusion.weight
        for abs_node in abs_nodes:
            abs_context = self._S[abs_node.context_id]
            # Calculate rule_weight (transition cost)
            # TODO Implement
            rule_weight = 0
            n_conclusion.priority += rule_weight + abs_context.weight
        if self._should_reexpand(n_conclusion):
            heap.heappush(self._Q, n_conclusion)

    def _should_reexpand(self, n_conclusion):
        if n_conclusion.abs_level == len(self._reexplore_levels):
            # This handles the START level of abstraction (highest)
            reexplore_level = ReexploreLevel.NONE
        else:
            reexplore_level = self._reexplore_levels[n_conclusion.abs_level]

        if reexplore_level == ReexploreLevel.FULL:
            return True
        elif reexplore_level == ReexploreLevel.PARTIAL:  # and (n_conclusion in S)
            return not (n_conclusion.id in self._S) or (
                n_conclusion.priority < self._S[n_conclusion.id].priority
            )
        elif reexplore_level == ReexploreLevel.NONE:
            return not (n_conclusion.id in self._S)

    def _update_vertex_visit_revisit(self, n: GCSHANode):
        if n.id in self._S:
            self._alg_metrics.n_vertices_revisited[n.abs_level][type(n).__name__] += 1
        else:
            self._alg_metrics.n_vertices_visited[n.abs_level][type(n).__name__] += 1
