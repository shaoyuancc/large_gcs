import heapq as heap
import itertools
import logging
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib.animation import FFMpegWriter

from large_gcs.algorithms.search_algorithm import (
    AlgMetrics,
    AlgVisParams,
    ReexploreLevel,
    SearchAlgorithm,
    TieBreak,
)
from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.graph.graph import Edge, Graph, ShortestPathSolution

logger = logging.getLogger(__name__)


class GcsAstarConvexRestriction(SearchAlgorithm):
    """Convex Restriction version of GCS A*, where in the subroutine, the order of vertices is fixed."""

    def __init__(
        self,
        graph: Graph,
        cost_estimator: CostEstimator,
        reexplore_level: ReexploreLevel = ReexploreLevel.NONE,
        tiebreak: TieBreak = TieBreak.LIFO,
        vis_params: AlgVisParams = AlgVisParams(),
    ):

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
        self._node_dists = defaultdict(lambda: float("inf"))
        self._visited = Graph(self._graph._default_costs_constraints)
        self._visited_vertices = set()
        if tiebreak == TieBreak.FIFO or tiebreak == TieBreak.FIFO.name:
            self._counter = itertools.count(start=0, step=1)
        elif tiebreak == TieBreak.LIFO or tiebreak == TieBreak.LIFO.name:
            self._counter = itertools.count(start=0, step=-1)
        # Ensures the source is the first node to be visited, even though the heuristic distance is not 0.
        heap.heappush(
            self._pq, (0, next(self._counter), self._graph.source_name, [], None)
        )

        # Add the target to the visited subgraph
        self._visited.add_vertex(
            self._graph.vertices[self._graph.target_name], self._graph.target_name
        )
        self._visited.set_target(self._graph.target_name)
        self.alg_metrics.n_vertices_visited = (
            1  # Start with the target node in the visited subgraph
        )

    def run(self, animate_intermediate: bool = False, final_plot: bool = False):
        logger.info(
            f"Running {self.__class__.__name__}, reexplore_level: {self._reexplore_level}"
        )
        self._animate_intermediate = animate_intermediate

        self._start_time = time.time()
        while self._candidate_sol is None and len(self._pq) > 0:
            self._run_iteration()
            self._alg_metrics.time_wall_clock = time.time() - self._start_time

        sol = self._candidate_sol
        if sol is None:
            logger.warn(f"Convex Restriction Gcs A* failed to find a solution.")
        else:
            logger.info(
                f"Convex Restriction Gcs A* complete! \ncost: {sol.cost}, time: {sol.time}\nvertex path: {np.array(sol.vertex_path)}\n{self.alg_metrics}"
            )
            self._graph._post_solve(sol)
        return sol

    def _run_iteration(self):
        # # Make a copy of the priority queue.
        # pq_copy = copy.copy(self._pq)
        # # Pop the top 10 items from the priority queue copy.
        # top_10 = [heap.heappop(pq_copy)[0] for _ in range(min(10, len(pq_copy)))]
        # logger.info(f"Top 10 pq costs: {top_10}")

        estimated_cost, _count, node, active_edges, contact_sol = heap.heappop(self._pq)
        if (
            self._reexplore_level == ReexploreLevel.NONE
            and node in self._visited_vertices
        ):
            return
        if node in self._visited_vertices:
            self._alg_metrics.n_vertices_revisited += 1
        else:
            self._alg_metrics.n_vertices_visited += 1

        self._visited_vertices.add(node)
        self._set_visited_vertices_and_edges(active_edges)

        edges = self._graph.outgoing_edges(node)

        logger.info(
            f"\n{self.alg_metrics}\nnow exploring node {node}'s {len(edges)} neighbors ({estimated_cost})"
        )
        if self._animate_intermediate and contact_sol is not None:
            self._graph.contact_spp_sol = contact_sol
            anim = self._graph.animate_solution()
            display(HTML(anim.to_html5_video()))

        if self._reexplore_level == ReexploreLevel.NONE:
            for edge in edges:
                if edge.v not in self._visited_vertices:
                    self._explore_edge(edge)
        else:
            for edge in edges:
                neighbor_in_path = any(
                    (e.u == edge.v or e.v == edge.v) for e in active_edges
                )
                if not neighbor_in_path:
                    self._explore_edge(edge)

    def _explore_edge(self, edge: Edge):
        neighbor = edge.v
        assert neighbor != self._graph.target_name
        if neighbor in self._visited_vertices:
            self._alg_metrics.n_vertices_reexplored += 1
        else:
            self._alg_metrics.n_vertices_explored += 1

        sol = self._cost_estimator.estimate_cost(
            self._visited, edge, solve_convex_restriction=True
        )

        if sol.is_success:
            new_dist = sol.cost
            should_add_to_pq = self._should_add_to_pq(neighbor, new_dist)
            logger.debug(
                f"edge {edge.u} -> {edge.v} is feasible, new dist: {new_dist}, added to pq {should_add_to_pq}"
            )
            if should_add_to_pq:
                new_active_edges = list(self._visited.edges.values()).copy() + [
                    self._graph.edges[edge.key]
                ]
                # Note this assumes graph is contact graph, should break this dependency...
                # Counter serves as tiebreaker for nodes with the same distance, to prevent nodes or edges from being compared
                heap.heappush(
                    self._pq,
                    (
                        new_dist,
                        next(self._counter),
                        neighbor,
                        new_active_edges,
                        self._graph.create_contact_spp_sol(
                            sol.vertex_path, sol.ambient_path
                        ),
                    ),
                )

            if new_dist < self._node_dists[neighbor]:
                self._node_dists[neighbor] = new_dist
                # Check if this neighbor actually has an edge to the target
                if (neighbor, self._graph.target_name) in self._graph.edges:
                    self._candidate_sol = sol

            if self._writer:
                self._writer.fig.clear()
                self.plot_graph(sol.ambient_path, edge)
                self._writer.grab_frame()

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
                and neighbor not in self._visited_vertices
            )

    def _set_visited_vertices_and_edges(self, edges):
        """Also adds source and target regardless of whether they are in edges"""
        self._visited = Graph(self._graph._default_costs_constraints)
        vertex_list = [self._graph.target_name, self._graph.source_name]
        for edge in edges:
            vertex_list.append(edge.v)
        for v in vertex_list:
            self._visited.add_vertex(self._graph.vertices[v], v)
        self._visited.set_source(self._graph.source_name)
        self._visited.set_target(self._graph.target_name)
        for edge in edges:
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
            round(self._node_dists[v], 1)
            if self._node_dists[v] != float("inf")
            else "∞"
            for v in self._graph.vertex_names
        ]
        self._graph.plot_set_labels(dist_labels)

        if path:
            if is_final_path:
                path_color = self._vis_params.final_path_color
            else:
                path_color = self._vis_params.intermediate_path_color
            self._graph.plot_path(path, color=path_color, linestyle="--")

    @property
    def alg_metrics(self):
        """Recompute metrics based on the current state of the algorithm.
        n_vertices_visited, n_gcs_solves, gcs_solve_time_total/min/max are manually updated.
        The rest are computed from the manually updated metrics.
        """
        return self._alg_metrics.update_derived_metrics(
            self._graph.n_vertices, self._graph.n_edges, self._visited.n_edges
        )
