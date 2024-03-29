import heapq as heap
import logging
import os
import time
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go

import wandb
from large_gcs.abstraction_models.abstraction_model import AbstractionModel
from large_gcs.abstraction_models.gcshastar_node import (
    ContextNode,
    GCSHANode,
    StatementNode,
)
from large_gcs.algorithms.gcs_astar_reachability import SetSamples
from large_gcs.algorithms.gcs_hastar import GcsHAstarMetrics
from large_gcs.algorithms.search_algorithm import (
    AlgMetrics,
    AlgVisParams,
    SearchAlgorithm,
)
from large_gcs.geometry.point import Point
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.graph import Edge, Graph, Vertex

logger = logging.getLogger(__name__)


class GcsHAstarReachability(SearchAlgorithm):
    def __init__(
        self,
        abs_model: AbstractionModel,
        vis_params: AlgVisParams = AlgVisParams(),
        num_samples_per_vertex: int = 100,
    ):
        self._graphs: List[ContactGraph] = abs_model.graphs
        self._abs_fns = abs_model.abs_fns
        self._vis_params = vis_params
        self._num_samples_per_vertex = num_samples_per_vertex

        # Expanded/Closed set
        self._S: dict[str, List[GCSHANode]] = defaultdict(list)
        self._S_ignored_counts: dict[str, int] = defaultdict(int)
        # Priority queue
        self._Q: list[GCSHANode] = []
        self._stalled: list[GCSHANode] = []  # Stalled nodes
        max_abs_level = len(self._graphs)
        self._alg_metrics = GcsHAstarMetrics()
        self._alg_metrics.initialize(n_levels=max_abs_level + 1)
        # Keeps track of samples for each vertex(set) in the graph.
        # These samples are not used directly but first projected into the feasible subspace of a particular path.
        self._set_samples: dict[str, SetSamples] = {}

        start_node = StatementNode.create_start_node(
            vertex_name="START", abs_level=max_abs_level, priority=0
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
                f"{self.__class__.__name__} failed to find a path to the target."
            )
            return

        g = self._graphs[0]
        g._post_solve(sol)
        logger.info(
            f"{self.__class__.__name__} complete! \ncost: {sol.cost}, time: {sol.time}"
            f"\nvertex path: {np.array(sol.vertex_path)}"
            f"\n{self.alg_metrics}"
        )
        return sol

    def _run_iteration(self):
        # self._iteration += 1
        # logger.info(f"\niteration: {self._iteration}")
        # logger.info(f"S {[(key, val.weight) for (key, val) in self._S.items()]}")
        # # Make a copy of the priority queue.
        # pq_copy = copy(self._Q)
        # # Pop the top 10 items from the priority queue copy.
        # bottom_10 = []
        # for _ in range(min(10, len(pq_copy))):
        #     n = heap.heappop(pq_copy)
        #     bottom_10.append((n.id, n.priority))
        # logger.info(f"Lowest 10 in Q: {bottom_10}")

        n: GCSHANode = heap.heappop(self._Q)

        logger.info(f"\n\n{self.alg_metrics}\nexpanding {n.id}, priority {n.priority}")

        if isinstance(n, StatementNode):
            # Check if BASE rule applies (the goal of a level is reached)
            if n.vertex_name == self._targets[n.abs_level]:
                self._update_vertex_expand_reexpand(n)
                logger.info(f"Goal reached at level {n.abs_level}")

                self._execute_base_rule(n)
                if n.abs_level == 0:
                    return n.sol
                else:
                    return None

            g = self._graphs[n.abs_level]
            if n.id in self._S:  # vertex has been expanded before
                # Check whether it reaches new regions
                if not self._reaches_new(g, n):
                    logger.info(
                        f"Not reexpanding: Path to {n.vertex_name} does not reach new samples"
                    )
                    self._S_ignored_counts[n.id] += 1
                    return
                else:
                    logger.info(
                        f"Reexpanding: Path to {n.vertex_name} reaches new samples"
                    )
            else:  # vertex has not been expanded before
                # Generate neighbors (vertices and edges) that you are about to explore
                g.generate_neighbors(n.vertex_name)

            self._update_vertex_expand_reexpand(n)

            # Get UP rules (edges at the same level)
            edges = g.outgoing_edges(n.vertex_name)
            for edge in edges:
                neighbor_in_path = any(
                    (g.edges[e].u == edge.v or g.edges[e].v == edge.v)
                    for e in n.edge_path
                )
                if not neighbor_in_path:
                    neighbor = StatementNode.from_parent(
                        child_vertex_name=edge.v, parent=n
                    )
                    abs_neighbor_nodes = self._abs_fns[n.abs_level](neighbor)
                    # Check if all required contexts of the abstracted neighbor are in S
                    if all(
                        [abs_n.context_id in self._S for abs_n in abs_neighbor_nodes]
                    ):
                        self._execute_up_rule(neighbor)
                    else:
                        logger.info(
                            f"required contexts for {neighbor.id} not in S , continuing"
                        )
                        self._stalled.append(neighbor)

        # CONTEXT and has other statements in path to convert to contexts
        elif isinstance(n, ContextNode):
            self._update_vertex_expand_reexpand(n)
            if len(n.edge_path) > 0:
                # Get DOWN rules
                # Go back one step along the path that was taken to calculate its context
                self._execute_down_rule(n_antecedent=n)
            else:
                # The context for the source node has been added to S, so we can now add the source node to Q
                self._execute_source_up_rule(n)

    def _execute_base_rule(self, n_antecedent: StatementNode):
        logger.info(f"Executing BASE rule for antecendent {n_antecedent.id}")
        # The antecedent will always be a goal node.

        # # EXPERIMENTAL: Try clearing statment nodes in lower level.
        # if self._reexplore_levels[n_antecedent.abs_level - 1] == ReexploreLevel.PARTIAL:
        #     count = 0
        #     for id, node in list(self._S.items()):
        #         if isinstance(node, StatementNode) and node.abs_level == n_antecedent.abs_level - 1:
        #             del self._S[id]
        #             count += 1
        #     logger.info(f"Cleared {count} statement nodes in level {n_antecedent.abs_level - 1}")

        # Extract path costs
        path_costs = []
        if len(n_antecedent.edge_path) > 0:
            original_g: ContactGraph = self._graphs[n_antecedent.abs_level]
            # WORKAROUND individual costs not being available in solve convex restriction result.
            # Create as new graph with just the path from the source to the goal and solve that as a full problem
            temp_g = Graph()
            temp_g.add_vertex(original_g.source, original_g.source_name)
            for v in n_antecedent.vertex_path[1:]:
                temp_g.add_vertex(original_g.vertices[v], v)
            for e_key in n_antecedent.edge_path:
                temp_g.add_edge(original_g.edges[e_key])
            temp_g.set_source(original_g.source_name)
            temp_g.set_target(original_g.target_name)
            sol = temp_g.solve_shortest_path()
            n_antecedent.sol = sol
            g = temp_g

            for e, v in zip(n_antecedent.edge_path, n_antecedent.vertex_path[1:]):
                vertex_cost = g.vertices[v].gcs_vertex.GetSolutionCost(
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
            edge_path=n_antecedent.edge_path,
            vertex_path=n_antecedent.vertex_path,
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
            vertex_name=n_antecedent.vertex_path[-2],
            # Weight of the prior context is parent's weight + vertex cost and edge cost (stored in path_costs)
            weight=n_antecedent.weight + n_antecedent.path_costs[-1],
            edge_path=n_antecedent.edge_path[:-1],
            vertex_path=n_antecedent.vertex_path[:-1],
            path_costs=n_antecedent.path_costs[:-1],
            sol=n_antecedent.sol,
            parent=n_antecedent,
        )

        self._update_vertex_visit_revisit(child)
        heap.heappush(self._Q, child)

    def _execute_source_up_rule(self, n_antecedent: ContextNode):
        logger.info(f"Executing SOURCE UP rule for antecedent {n_antecedent.id}")

        lower_abs_level = n_antecedent.abs_level - 1

        # EXPERIMENTAL: Try executing stalled nodes in lower level
        n_stalled = len(self._stalled)
        if n_stalled > 0:
            still_stalled = []
            for neighbor in self._stalled:
                abs_neighbor_nodes = self._abs_fns[lower_abs_level](neighbor)
                if neighbor.abs_level == lower_abs_level and all(
                    [
                        abs_n.context_id in self._S for abs_n in abs_neighbor_nodes
                    ]  # Check if all required contexts of the abstracted neighbor are in S
                ):
                    self._execute_up_rule(neighbor)
                else:
                    still_stalled.append(neighbor)
            self._stalled = still_stalled
            logger.info(
                f"Pushed {n_stalled - len(self._stalled)} stalled nodes in level {lower_abs_level} back into Q, {len(self._stalled)} still stalled"
            )

        source_name = self._graphs[lower_abs_level].source_name
        # add the source node of the next level to the queue
        n_source = StatementNode.create_start_node(
            vertex_name=source_name,
            abs_level=lower_abs_level,
            # We want this to get popped off immediately after the context above is popped off
            priority=n_antecedent.priority,
        )

        self._update_vertex_visit_revisit(n_source)
        heap.heappush(self._Q, n_source)

    def _execute_up_rule(self, n_conclusion: GCSHANode):
        """UP rule: visit neighbor"""
        logger.debug(f"Executing UP rule for conclusion {n_conclusion.id}")

        self._update_vertex_visit_revisit(n_conclusion)

        g: ContactGraph = self._graphs[n_conclusion.abs_level]
        abs_fn = self._abs_fns[n_conclusion.abs_level]

        # Solve convex restriction on the path from the source to the conclusion to get the weight
        g.set_target(n_conclusion.vertex_name)
        sol = g.solve_convex_restriction(n_conclusion.edge_path)
        self._alg_metrics.update_after_gcs_solve(sol.time)
        g.set_target(self._targets[n_conclusion.abs_level])
        if not sol.is_success:
            logger.debug(
                f"edge {n_conclusion.parent.vertex_name} -> {n_conclusion.vertex_name} not actually feasible"
            )
            # Conclusion invalid, do nothing, don't add to Q
            return
        else:
            logger.debug(
                f"edge {n_conclusion.parent.vertex_name} -> {n_conclusion.vertex_name} is feasible"
            )
        n_conclusion.sol = sol
        n_conclusion.weight = sol.cost
        abs_nodes: List[StatementNode] = abs_fn(n_conclusion)
        n_conclusion.priority = n_conclusion.weight
        for abs_node in abs_nodes:
            # TODO which context should be used? For now using the last one that was added
            # This is definitely wrong, but it should work for now
            # This choice probably has implications on optimality, correctness, completeness
            abs_context = self._S[abs_node.context_id][-1]
            # Calculate rule_weight (transition cost)
            # TODO Implement
            rule_weight = 0
            n_conclusion.priority += rule_weight + abs_context.weight
            # logger.debug(
            #     f"priority: {n_conclusion.priority} sol_cost: {sol.cost}, abs_context_weight: {abs_context.weight}"
            # )

        heap.heappush(self._Q, n_conclusion)
        logger.debug(
            f"Added {n_conclusion.id} to Q with priority {n_conclusion.priority}"
        )

    def _update_vertex_visit_revisit(self, n: GCSHANode):
        if n.id in self._S:
            self._alg_metrics.n_vertices_revisited[n.abs_level][type(n).__name__] += 1
        else:
            self._alg_metrics.n_vertices_visited[n.abs_level][type(n).__name__] += 1

    def _update_vertex_expand_reexpand(self, n: GCSHANode):
        if n.id in self._S:
            self._alg_metrics.n_vertices_reexpanded[n.abs_level][type(n).__name__] += 1
        else:
            self._alg_metrics.n_vertices_expanded[n.abs_level][type(n).__name__] += 1

        self._S[n.id] += [n]
        self.log_metrics_to_wandb(n.priority)

    def _reaches_new(self, g: Graph, n: StatementNode) -> bool:
        """
        Checks samples to see if this path reaches new previously unreached samples.
        Assumes that this path is feasible and that the vertex has been expanded before.
        i.e. there already exists a path that reaches the vertex.
        """
        # Assume that entire source is already reachable
        if n.vertex_name == g.source_name:
            return False

        if n.vertex_name not in self._set_samples:
            logger.debug(f"Adding samples for {n.vertex_name}")
            self._set_samples[n.vertex_name] = SetSamples.from_vertex(
                n.vertex_name,
                g.vertices[n.vertex_name],
                self._num_samples_per_vertex,
            )

        projected_samples = self._set_samples[n.vertex_name].project_all(g, n)
        reached_new = False
        for idx, sample in enumerate(projected_samples):
            # Create a new vertex for the sample and add it to the graph
            sample_vertex_name = f"{n.vertex_name}_sample_{idx}"
            g.add_vertex(
                vertex=Vertex(convex_set=Point(sample)), name=sample_vertex_name
            )
            go_to_next_sample = False
            for alt_n in self._S[n.id]:
                # Add edge between the sample and the second last vertex in the path
                e = g.edges[alt_n.edge_path[-1]]
                edge_to_sample = Edge(
                    u=e.u,
                    v=sample_vertex_name,
                    costs=e.costs,
                    constraints=e.constraints,
                )
                g.add_edge(edge_to_sample)
                # Check whether sample can be reached via the path
                g.set_target(sample_vertex_name)
                active_edges = alt_n.edge_path.copy()
                active_edges[-1] = edge_to_sample.key

                sol = g.solve_convex_restriction(active_edges, skip_post_solve=True)
                self._alg_metrics.update_after_gcs_solve(sol.time)
                if sol.is_success:
                    # Clean up current sample
                    g.remove_vertex(sample_vertex_name)
                    # Move on to the next sample, don't need to check other paths
                    logger.debug(f"Sample {idx} reached by path {alt_n.vertex_path}")
                    go_to_next_sample = True
                    break
                else:
                    # Clean up edge, but leave the sample vertex
                    g.remove_edge(edge_to_sample.key)
            if go_to_next_sample:
                continue
            # If no paths can reach the sample, do not need to check more samples
            reached_new = True
            # Clean up
            if sample_vertex_name in g.vertices:
                g.remove_vertex(sample_vertex_name)

            logger.debug(f"Sample {idx} not reached by any previous path.")
            break
        g.set_target(self._targets[n.abs_level])
        return reached_new

    def log_metrics_to_wandb(self, total_estimated_cost: float):
        if not self._S and wandb.run is not None:
            wandb.log(
                {
                    "total_estimated_cost": total_estimated_cost,
                    "alg_metrics": self.alg_metrics.to_dict(),
                }
            )
            return
        if self._vis_params.log_dir is not None:
            # Preparing tracked and ignored counts
            tracked_counts = [len(self._S[id]) for id in self._S]
            ignored_counts = [
                self._S_ignored_counts[id] for id in self._S_ignored_counts
            ]

            # Create a figure to plot the histograms
            fig = self._alg_metrics.create_paths_per_vertex_hist(
                tracked_counts, ignored_counts
            )

            # Save the figure to a file as png
            fig.write_image(
                os.path.join(self._vis_params.log_dir, "paths_per_vertex_hist.png")
            )

            if wandb.run is not None:
                # Log the Plotly figure and other metrics to wandb
                wandb.log(
                    {
                        "total_estimated_cost": total_estimated_cost,
                        "alg_metrics": self.alg_metrics.to_dict(),
                        "paths_per_vertex_hist": wandb.Plotly(fig),
                    }
                )
