import heapq as heap
import itertools
import logging
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from large_gcs.algorithms.search_algorithm import (
    AlgMetrics,
    AlgVisParams,
    ReexploreLevel,
    SearchAlgorithm,
    TieBreak,
)
from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.graph.graph import Edge, Graph

logger = logging.getLogger(__name__)


class GcsAstarTree(SearchAlgorithm):
    """Tree version of GCS A*, where we only add edges along shortest paths to
    vertices to the visited subgraph."""

    def __init__(
        self,
        graph: Graph,
        cost_estimator: CostEstimator,
        reexplore_level: ReexploreLevel = ReexploreLevel.NONE,
        tiebreak: TieBreak = TieBreak.LIFO,
        vis_params: AlgVisParams = AlgVisParams(),
    ):
        logger.warning("This class is deprecated and broken.")
        self._graph = graph
        self._cost_estimator = cost_estimator
        self._reexplore_level = (
            ReexploreLevel[reexplore_level]
            if type(reexplore_level) == str
            else reexplore_level
        )
        self._vis_params = vis_params
        self._writer = None
        self._alg_metrics = AlgMetrics()
        self._cost_estimator.set_alg_metrics(self._alg_metrics)
        self._candidate_sol = None
        self._pq = []
        self._shortest_path_edges = set()
        self._node_dists = defaultdict(lambda: float("inf"))
        self._visited = Graph(self._graph._default_costs_constraints)
        if tiebreak == TieBreak.FIFO or tiebreak == TieBreak.FIFO.name:
            self._counter = itertools.count(start=0, step=1)
        elif tiebreak == TieBreak.LIFO or tiebreak == TieBreak.LIFO.name:
            self._counter = itertools.count(start=0, step=-1)
        # Ensures the source is the first node to be visited, even though the heuristic distance is not 0.
        heap.heappush(self._pq, (0, next(self._counter), self._graph.source_name))

        # Add the target to the visited subgraph
        self._visited.add_vertex(
            self._graph.vertices[self._graph.target_name], self._graph.target_name
        )
        self._visited.set_target(self._graph.target_name)
        # Start with the target node in the visited subgraph
        self.alg_metrics.n_vertices_expanded[0] = 1

    def run(self):
        logger.info(
            f"Running {self.__class__.__name__}, reexplore_level: {self._reexplore_level}"
        )

        self._start_time = time.time()
        while len(self._pq) > 0:
            # Check for termination condition
            curr = self._pq[0][2]
            if (
                self._graph.target_name
                in [e.v for e in self._graph.outgoing_edges(curr)]
                and (curr, self._graph.target_name) in self._shortest_path_edges
            ):
                sol = self._candidate_sol
                self._graph._post_solve(sol)
                logger.info(
                    f"Gcs A* Tree complete! \ncost: {sol.cost}, time: {sol.time}\nvertex path: {np.array(sol.vertex_path)}\n{self.alg_metrics}"
                )

                return sol

            self._run_iteration()
            self._alg_metrics.time_wall_clock = time.time() - self._start_time

        logger.warning("Gcs A* Tree failed to find a path to the target.")
        return None

    def _run_iteration(self):
        estimated_cost, _count, node = heap.heappop(self._pq)
        if (
            self._reexplore_level == ReexploreLevel.NONE
            and node in self._visited.vertices
        ):
            return
        if node in self._visited.vertices:
            self._alg_metrics.n_vertices_reexpanded[0] += 1
        else:
            self._alg_metrics.n_vertices_expanded[0] += 1

        self._add_vertex_and_edges_to_visited_except_edges_to_target(node)

        edges = self._graph.outgoing_edges(node)

        logger.info(
            f"\n{self.alg_metrics}\nnow exploring node {node}'s {len(edges)} neighbors ({estimated_cost})"
        )

        if self._reexplore_level == ReexploreLevel.NONE:
            for edge in edges:
                if edge.v not in self._visited.vertices:
                    self._explore_edge(edge)
        else:
            for edge in edges:
                self._explore_edge(edge)

    def _explore_edge(self, edge: Edge):
        neighbor = edge.v
        assert neighbor != self._graph.target_name
        if neighbor in self._visited.vertices:
            self._alg_metrics.n_vertices_revisited[0] += 1
        else:
            self._alg_metrics.n_vertices_visited[0] += 1

        sol = self._cost_estimator.estimate_cost(
            self._visited, edge, solve_convex_restriction=False
        )

        if sol.is_success:
            new_dist = sol.cost
            should_add_to_pq = self._should_add_to_pq(neighbor, new_dist)
            logger.debug(
                f"edge {edge.u} -> {edge.v} is feasible, new dist: {new_dist}, added to pq {should_add_to_pq}"
            )
            if should_add_to_pq:
                # Note this assumes graph is contact graph, should break this dependency...
                # Counter serves as tiebreaker for nodes with the same distance, to prevent nodes or edges from being compared
                heap.heappush(
                    self._pq,
                    (
                        new_dist,
                        next(self._counter),
                        neighbor,
                    ),
                )

            if new_dist < self._node_dists[neighbor]:
                self._node_dists[neighbor] = new_dist
                self._shortest_path_edges.add((edge.u, edge.v))
                # Check if this neighbor actually has an edge to the target
                if (neighbor, self._graph.target_name) in self._graph.edges:
                    self._shortest_path_edges.add((neighbor, self._graph.target_name))
                    if self._candidate_sol is None:
                        self._candidate_sol = sol
                    elif sol.cost < self._candidate_sol.cost:
                        self._candidate_sol = sol

        else:
            logger.debug(f"edge {edge.u} -> {edge.v} not actually feasible")

    def _should_add_to_pq(self, neighbor, new_dist):
        if self._reexplore_level == ReexploreLevel.FULL:
            return True
        elif self._reexplore_level == ReexploreLevel.PARTIAL:
            return new_dist < self._node_dists[neighbor]
        elif self._reexplore_level == ReexploreLevel.NONE:
            return (
                new_dist < self._node_dists[neighbor]
                and neighbor not in self._visited.vertices
            )

    def _add_vertex_and_edges_to_visited_except_edges_to_target(self, vertex_name):
        """Also adds source and target regardless of whether they are in
        edges."""
        self._visited.add_vertex(self._graph.vertices[vertex_name], vertex_name)
        if vertex_name == self._graph.source_name:
            self._visited.set_source(self._graph.source_name)
        edges = self._graph.incident_edges(vertex_name)
        for edge in edges:
            if (
                edge.u in self._visited.vertex_names
                and edge.v in self._visited.vertex_names
                and edge.v != self._graph.target_name
                # Crucial part of this alg, only add edges that are on the shortest path
                and (edge.u, edge.v) in self._shortest_path_edges
            ):
                self._visited.add_edge(edge)

    def plot_graph(self, path=None, current_edge=None, is_final_path=False):
        plt.title("GCS A*")
        if self._graph.workspace is not None:
            plt.xlim(self._graph.workspace[0])
            plt.ylim(self._graph.workspace[1])
        plt.gca().set_aspect("equal")
        for vertex_name, vertex in self._graph.vertices.items():
            if current_edge and vertex_name == current_edge.u:
                vertex.convex_set.plot(facecolor=self._vis_params.relaxing_from_color)
            elif current_edge and vertex_name == current_edge.v:
                vertex.convex_set.plot(facecolor=self._vis_params.relaxing_to_color)
            elif vertex_name in self._visited.vertex_names:
                vertex.convex_set.plot(facecolor=self._vis_params.visited_vertex_color)
            elif vertex_name in [item[1] for item in self._pq]:
                vertex.convex_set.plot(facecolor=self._vis_params.frontier_color)
            else:
                vertex.convex_set.plot()
        for edge_key in self._graph.edge_keys:
            if (
                current_edge
                and edge_key[0] == current_edge.u
                and edge_key[1] == current_edge.v
            ):
                self._graph.plot_edge(
                    edge_key, color=self._vis_params.relaxing_edge_color, zorder=3
                )
            elif edge_key in self._visited.edge_keys:
                self._graph.plot_edge(
                    edge_key, color=self._vis_params.visited_edge_color
                )
            else:
                self._graph.plot_edge(edge_key, color=self._vis_params.edge_color)
        dist_labels = [
            (
                round(self._node_dists[v], 1)
                if self._node_dists[v] != float("inf")
                else "∞"
            )
            for v in self._graph.vertex_names
        ]
        self._graph.plot_set_labels(dist_labels)

        if path:
            if is_final_path:
                path_color = self._vis_params.final_path_color
            else:
                path_color = self._vis_params.intermediate_path_color
            self._graph.plot_path(path, color=path_color, linestyle="--")
