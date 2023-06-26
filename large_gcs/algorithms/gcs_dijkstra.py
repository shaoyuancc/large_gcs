from collections import defaultdict
import numpy as np
import heapq as heap
from IPython.display import clear_output
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
from large_gcs.algorithms.search_algorithm import (
    AlgMetrics,
    SearchAlgorithm,
    AlgVisParams,
)
from large_gcs.graph.graph import Graph

# matplotlib.use("Agg")


class GcsDijkstra(SearchAlgorithm):
    def __init__(self, graph: Graph, vis_params: AlgVisParams = AlgVisParams()):
        self._graph = graph
        self._vis_params = vis_params
        self._writer = None
        self._alg_metrics = AlgMetrics()
        self._gcs_solve_times = np.empty((0,))
        self._pq = []
        self._node_dists = defaultdict(lambda: float("inf"))
        self._visited = Graph()
        self._node_dists[self._graph.source_name] = 0
        heap.heappush(self._pq, (0, self._graph.source_name))

    def run(self, animate: bool = False):
        if animate:
            metadata = dict(title="GCS Dijkstra", artist="Matplotlib")
            self._writer = FFMpegWriter(fps=self._vis_params.fps, metadata=metadata)
            fig = plt.figure(figsize=self._vis_params.figsize)
            self._writer.setup(fig, self._vis_params.output_path, self._vis_params.dpi)

        while len(self._pq) > 0 and self._pq[0][1] != self._graph.target_name:
            self._run_iteration()

        # Solve GCS for a final time to extract the path
        self._add_vertex_and_edges_to_visited(self._graph.target_name)
        self._visited.set_target(self._graph.target_name)
        sol = self._visited.solve_shortest_path()
        self._update_alg_metrics_after_gcs_solve(sol.time)
        print(f"Gcs Dijkstra complete! \n{sol}\n{self.alg_metrics}")
        if self._writer:
            self._writer.fig.clear()
            self.plot_graph(path=sol.path)
            self._writer.grab_frame()
            self._writer.finish()
            self._writer = None
        return sol

    def _run_iteration(self):
        _, node = heap.heappop(self._pq)
        if node in self._visited.vertex_names:
            return

        self._add_vertex_and_edges_to_visited(node)

        if node == self._graph.source_name:
            self._visited.set_source(self._graph.source_name)

        if self._writer:
            clear_output(wait=True)
            print(f"{self.alg_metrics}, now relaxing node {node}")
            self._writer.fig.clear()
            self.plot_graph()
            self._writer.grab_frame()

        edges = self._graph.outgoing_edges(node)
        for edge in edges:
            neighbor = edge.v
            if neighbor not in self._visited.vertex_names:
                # Add neighbor and edge temporarily to the visited subgraph
                self._visited.add_vertex(self._graph.vertices[neighbor], neighbor)
                self._visited.add_edge(edge)
                self._visited.set_target(neighbor)

                sol = self._visited.solve_shortest_path()
                new_dist = sol.cost

                # Remove neighbor and associated edges from the visited subgraph
                self._visited.remove_vertex(neighbor)

                self._update_alg_metrics_after_gcs_solve(sol.time)

                if new_dist < self._node_dists[neighbor]:
                    self._node_dists[neighbor] = new_dist
                    heap.heappush(self._pq, (new_dist, neighbor))

                if self._writer:
                    self._writer.fig.clear()
                    self.plot_graph(sol.path, edge)
                    self._writer.grab_frame()

    def _add_vertex_and_edges_to_visited(self, vertex_name):
        # Add node to the visited subgraph along with all of its incoming and outgoing edges to the visited subgraph
        self._visited.add_vertex(self._graph.vertices[vertex_name], vertex_name)
        self._alg_metrics.n_vertices_visited += 1
        edges = self._graph.incident_edges(vertex_name)
        for edge in edges:
            if (
                edge.u in self._visited.vertex_names
                and edge.v in self._visited.vertex_names
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

    def plot_graph(self, path=None, current_edge=None):
        plt.title("GCS Dijkstra")
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
                vertex.convex_set.plot(facecolor=self._vis_params.visited_color)
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
            if self._graph._target_name in self._visited.vertex_names:
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
