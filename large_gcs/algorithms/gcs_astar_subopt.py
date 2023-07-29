import copy
import heapq as heap
import itertools
import logging
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from large_gcs.algorithms.search_algorithm import (
    AlgMetrics,
    AlgVisParams,
    SearchAlgorithm,
)
from large_gcs.cost_estimators.shortcut_edge_ce import ShortcutEdgeCE
from large_gcs.graph.graph import Edge, Graph

logger = logging.getLogger(__name__)


class GcsAstarSubOpt(SearchAlgorithm):
    """Suboptimal version of GCS A*. This algorithm is not complete, a solution is not guaranteed to be found even if it exists."""

    def __init__(
        self,
        graph: Graph,
        use_convex_relaxation: bool = False,
        shortcut_edge_cost_factory=None,
        vis_params: AlgVisParams = AlgVisParams(),
    ):
        if (
            shortcut_edge_cost_factory is None
            and graph._default_costs_constraints.edge_costs is None
        ):
            raise ValueError(
                "If no shortcut_edge_cost_factory is specified, edge costs must be specified in the graph's default costs constraints."
            )

        self._graph = graph
        self._use_convex_relaxation = use_convex_relaxation
        self._shortcut_edge_cost_factory = shortcut_edge_cost_factory
        self._vis_params = vis_params
        self._writer = None
        self._alg_metrics = AlgMetrics()
        self._gcs_solve_times = np.empty((0,))
        self._candidate_sol = None
        self._pq = []
        self._feasible_edges = set()
        self._node_dists = defaultdict(lambda: float("inf"))
        self._visited = Graph(self._graph._default_costs_constraints)
        self._counter = itertools.count(start=0, step=1)
        self._cost_estimator = ShortcutEdgeCE(graph, shortcut_edge_cost_factory)
        self._cost_estimator.set_alg_metrics(self._alg_metrics)
        # Ensures the source is the first node to be visited, even though the heuristic distance is not 0.
        heap.heappush(self._pq, (0, next(self._counter), self._graph.source_name))

        # Add the target to the visited subgraph
        self._visited.add_vertex(
            self._graph.vertices[self._graph.target_name], self._graph.target_name
        )
        self._visited.set_target(self._graph.target_name)
        self.alg_metrics.n_vertices_visited = (
            1  # Start with the target node in the visited subgraph
        )

    def run(
        self, verbose: bool = False, animate: bool = False, final_plot: bool = False
    ):
        if animate:
            metadata = dict(title="GCS A*", artist="Matplotlib")
            self._writer = FFMpegWriter(fps=self._vis_params.fps, metadata=metadata)
            fig = plt.figure(figsize=self._vis_params.figsize)
            self._writer.setup(
                fig, self._vis_params.vid_output_path, self._vis_params.dpi
            )
        self._start_time = time.time()
        while self._candidate_sol is None and len(self._pq) > 0:
            self._run_iteration(verbose=verbose)
            self._alg_metrics.time_wall_clock = time.time() - self._start_time

        sol = self._candidate_sol
        if sol is None:
            print(f"No solution found! {self.alg_metrics}")
        else:
            # clear_output(wait=True)
            print(
                f"Suboptimal Gcs A* complete! \ncost: {sol.cost}, time: {sol.time}\nvertex path: {np.array(sol.vertex_path)}\n{self.alg_metrics}"
            )
            if self._writer:
                self._writer.fig.clear()
                self.plot_graph(path=sol.ambient_path, is_final_path=True)
                self._writer.grab_frame()
                self._writer.finish()
                self._writer = None
            if final_plot:
                if not animate:
                    fig = plt.figure(figsize=self._vis_params.figsize)
                    self.plot_graph(path=sol.ambient_path, is_final_path=True)
                plt.savefig(self._vis_params.plot_output_path)
                plt.show()
        return sol

    def _run_iteration(self, verbose: bool = False):
        # Make a copy of the priority queue.
        pq_copy = copy.copy(self._pq)
        # Pop the top 10 items from the priority queue copy.
        top_10 = [heap.heappop(pq_copy)[0] for _ in range(min(10, len(pq_copy)))]
        logger.info(f"Top 10 pq costs: {top_10}")

        estimated_cost, _count, node = heap.heappop(self._pq)
        if node in self._visited.vertex_names and node != self._graph.target_name:
            return

        self._add_vertex_and_edges_to_visited_except_edges_to_target(node)

        if node == self._graph.source_name:
            self._visited.set_source(self._graph.source_name)

        edges = self._graph.outgoing_edges(node)

        logger.info(
            f"\n{self.alg_metrics}\nnow exploring node {node}'s {len(edges)} neighbors ({estimated_cost})"
        )
        if self._writer:
            self._writer.fig.clear()
            self.plot_graph()
            self._writer.grab_frame()

        for edge in edges:
            neighbor = edge.v
            if neighbor not in self._visited.vertex_names:
                self._explore_edge(edge, verbose=verbose)

    def _explore_edge(self, edge: Edge, verbose: bool = False):
        self._alg_metrics.n_vertices_explored += 1
        neighbor = edge.v
        assert neighbor != self._graph.target_name
        """
        Very strange, there is something about using the cost estimator instead of 
        the code below that changes the numbers very slightly.
        """
        # # Add neighbor and edge temporarily to the visited subgraph
        # self._visited.add_vertex(self._graph.vertices[neighbor], neighbor)
        # # Check if this neighbor actually has an edge to the target
        # # If so, add that edge instead of the shortcut
        # if (neighbor, self._graph.target_name) in self._graph.edges:
        #     self._visited.add_edge(
        #         self._graph.edges[(neighbor, self._graph.target_name)]
        #     )
        # else:
        #     # Add an edge from the neighbor to the target
        #     direct_edge_costs = None
        #     if self._shortcut_edge_cost_factory:
        #         # Note for now this only works with ContactSet and ContactPointSet because
        #         # they have the vars attribute, and convex_sets in general do not.
        #         direct_edge_costs = self._shortcut_edge_cost_factory(
        #             self._graph.vertices[neighbor].convex_set.vars,
        #             self._graph.vertices[self._graph.target_name].convex_set.vars,
        #         )
        #     direct_to_target = Edge(
        #         neighbor, self._graph.target_name, costs=direct_edge_costs
        #     )
        #     self._visited.add_edge(direct_to_target)

        # self._visited.add_edge(edge)
        # sol = self._visited.solve(self._use_convex_relaxation)

        # self._update_alg_metrics_after_gcs_solve(sol.time)

        # # Remove neighbor and associated edges from the visited subgraph
        # self._visited.remove_vertex(neighbor)
        sol = self._cost_estimator.estimate_cost(
            self._visited, edge, solve_convex_restriction=False
        )

        if sol.is_success:
            new_dist = sol.cost
            # if verbose:
            #     print(
            #         f"edge {edge.u} -> {edge.v} is feasible, new dist: {new_dist}, added to pq {new_dist < self._node_dists[neighbor]}"
            #     )
            if (
                new_dist < self._node_dists[neighbor]
                and neighbor not in self._visited.vertices
            ):
                heap.heappush(self._pq, (new_dist, next(self._counter), neighbor))

            if new_dist < self._node_dists[neighbor]:
                self._feasible_edges.add((edge.u, edge.v))
                self._node_dists[neighbor] = new_dist

                # Check if this neighbor actually has an edge to the target
                if (neighbor, self._graph.target_name) in self._graph.edges:
                    self._candidate_sol = sol

            if self._writer:
                self._writer.fig.clear()
                self.plot_graph(sol.ambient_path, edge)
                self._writer.grab_frame()

        else:
            pass
            # if verbose:
            #     print(f"edge {edge.u} -> {edge.v} not actually feasible")

    def _add_vertex_and_edges_to_visited_except_edges_to_target(self, vertex_name):
        # Add node to the visited subgraph along with all of its incoming and outgoing edges to the visited subgraph
        self._visited.add_vertex(self._graph.vertices[vertex_name], vertex_name)
        self._alg_metrics.n_vertices_visited += 1
        edges = self._graph.incident_edges(vertex_name)
        for edge in edges:
            if (
                edge.u in self._visited.vertex_names
                and edge.v in self._visited.vertex_names
                and edge.v != self._graph.target_name
                and (edge.u, edge.v)
                in self._feasible_edges  # Experimental, only add edges we've verified are feasible
            ):
                self._visited.add_edge(edge)

    def _update_alg_metrics_after_gcs_solve(self, solve_time: float):
        m = self._alg_metrics
        m.n_gcs_solves += 1
        m.gcs_solve_time_total += solve_time

        if solve_time < m.gcs_solve_time_iter_min:
            m.gcs_solve_time_iter_min = solve_time
        if solve_time > m.gcs_solve_time_iter_max:
            m.gcs_solve_time_iter_max = solve_time
        self._gcs_solve_times = np.append(self._gcs_solve_times, solve_time)

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
        m = self._alg_metrics
        m.vertex_coverage = round(m.n_vertices_visited / self._graph.n_vertices, 2)
        m.n_edges_visited = self._visited.n_edges
        m.edge_coverage = round(m.n_edges_visited / self._graph.n_edges, 2)
        if m.n_gcs_solves > 0:
            m.gcs_solve_time_iter_mean = m.gcs_solve_time_total / m.n_gcs_solves
            m.gcs_solve_time_iter_std = np.std(self._gcs_solve_times)
        return m
