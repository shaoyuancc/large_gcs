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
    TieBreak,
)
from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.graph.graph import Edge, Graph

logger = logging.getLogger(__name__)


class GcsAstar(SearchAlgorithm):
    """A* search algorithm for GCS. This implementation is very similar to GCS Dijkstra,
    but the the target node is always added to the visited subgraph, and an edge from the
    node being explored to the target is also added.
    Current implementation requires heuristic to be both admissible and consistent.
    Currently does not handle infeasible edges correctly, thus this algorithm is not complete.
    """

    def __init__(
        self,
        graph: Graph,
        cost_estimator: CostEstimator,
        use_convex_relaxation: bool = False,
        tiebreak: TieBreak = TieBreak.LIFO,
        vis_params: AlgVisParams = AlgVisParams(),
    ):
        self._graph = graph
        self._cost_estimator = cost_estimator
        self._use_convex_relaxation = use_convex_relaxation
        self._vis_params = vis_params
        self._writer = None
        self._alg_metrics = AlgMetrics()
        self._cost_estimator.set_alg_metrics(self._alg_metrics)
        self._candidate_sol = None
        self._pq = []
        self._feasible_edges = set()
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
        # Add the source to the visited subgraph
        self._visited.add_vertex(
            self._graph.vertices[self._graph.source_name], self._graph.source_name
        )
        self._visited.set_source(self._graph.source_name)
        self._visited.set_target(self._graph.target_name)
        self.alg_metrics.n_vertices_visited = (
            2  # Start with the source and target node in the visited subgraph
        )

        self._cost_estimator.setup_subgraph(self._visited)

    def run(self, animate: bool = False, final_plot: bool = False):
        if animate:
            metadata = dict(title="GCS A*", artist="Matplotlib")
            self._writer = FFMpegWriter(fps=self._vis_params.fps, metadata=metadata)
            fig = plt.figure(figsize=self._vis_params.figsize)
            self._writer.setup(
                fig, self._vis_params.vid_output_path, self._vis_params.dpi
            )
        self._start_time = time.time()
        while len(self._pq) > 0:
            # Check for termination condition
            curr = self._pq[0][2]
            if curr == self._graph.target_name:
                sol = self._candidate_sol
                self._graph._post_solve(sol)
                logger.info(
                    f"Gcs A* complete! \ncost: {sol.cost}, time: {sol.time}\nvertex path: {np.array(sol.vertex_path)}\n{self.alg_metrics}"
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

            self._run_iteration()
            self._alg_metrics.time_wall_clock = time.time() - self._start_time

        logger.warn("Gcs A* failed to find a path to the target.")
        return None

    def _run_iteration(self):
        estimated_cost, _count, node = heap.heappop(self._pq)
        if node in self._visited.vertex_names and node != self._graph.source_name:
            return

        self._add_vertex_and_edges_to_visited_except_edges_to_target(node)

        edges = self._graph.outgoing_edges(node)

        logger.info(
            f"\n{self.alg_metrics}\nnow exploring node {node}'s {len(edges)} neighbors ({estimated_cost})"
        )
        self.log_metrics_to_wandb(estimated_cost)

        if self._writer:
            self._writer.fig.clear()
            self.plot_graph()
            self._writer.grab_frame()

        for edge in edges:
            if (
                edge.v not in self._visited.vertex_names
                or edge.v == self._graph.target_name
            ):
                self._alg_metrics.n_vertices_explored += 1
                self._explore_edge(edge)

    def _explore_edge(self, edge: Edge):
        neighbor = edge.v

        sol = self._cost_estimator.estimate_cost(
            self._visited, edge, use_convex_relaxation=self._use_convex_relaxation
        )

        if sol.is_success:
            new_dist = sol.cost
            logger.debug(
                f"edge {edge.u} -> {edge.v} is feasible, new dist: {new_dist}, added to pq {new_dist < self._node_dists[neighbor]}"
            )
            if new_dist < self._node_dists[neighbor]:
                self._node_dists[neighbor] = new_dist
                heap.heappush(self._pq, (new_dist, next(self._counter), neighbor))
                # Check if this neighbor is actually the target
                if neighbor == self._graph.target_name:
                    if self._candidate_sol is None:
                        self._candidate_sol = sol
                    elif sol.cost < self._candidate_sol.cost:
                        self._candidate_sol = sol

            if self._writer:
                self._writer.fig.clear()
                self.plot_graph(sol.ambient_path, edge)
                self._writer.grab_frame()
            self._feasible_edges.add((edge.u, edge.v))
        else:
            logger.debug(f"edge {edge.u} -> {edge.v} not actually feasible")

    def _add_vertex_and_edges_to_visited_except_edges_to_target(self, vertex_name):
        # Add node to the visited subgraph along with all of its incoming and outgoing edges to the visited subgraph
        self._graph.generate_neighbors(vertex_name)

        if vertex_name in self._visited.vertex_names:
            return

        self._visited.add_vertex(self._graph.vertices[vertex_name], vertex_name)
        self._alg_metrics.n_vertices_visited += 1
        edges = self._graph.incident_edges(vertex_name)
        for edge in edges:
            if (
                edge.u in self._visited.vertex_names
                and edge.v in self._visited.vertex_names
                and edge.v != self._graph.target_name
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
