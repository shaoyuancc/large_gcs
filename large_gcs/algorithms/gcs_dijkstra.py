from collections import defaultdict
import heapq as heap
import matplotlib
from matplotlib import animation
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
from large_gcs.algorithms.search_algorithm import SearchAlgorithm, AlgVisParams
from large_gcs.graph.graph import Graph

matplotlib.use("Agg")


class GcsDijkstra(SearchAlgorithm):
    def __init__(self, graph: Graph, vis_params: AlgVisParams = AlgVisParams()):
        self._graph = graph
        self._vis_params = vis_params
        self._writer = None
        self.iteration_count = 0
        self._pq = []
        self._node_dists = defaultdict(lambda: float("inf"))
        self._visited = Graph()
        self._node_dists[self._graph.source_name] = 0
        heap.heappush(self._pq, (0, self._graph.source_name))

    def run(self, animate: bool = False):
        if animate:
            metadata = dict(title="GCS Dijkstra", artist="Matplotlib")
            self._writer = FFMpegWriter(fps=self._vis_params.fps, metadata=metadata)
            fig = plt.figure()
            self._writer.setup(fig, self._vis_params.output_path, self._vis_params.dpi)

        while len(self._pq) > 0:
            self._run_iteration(animate)

        # Solve GCS for a final time to extract the path
        self._visited.set_target(self._graph.target_name)
        sol = self._visited.solve_shortest_path()
        print(f"Cost: {sol.cost}")
        if self._writer:
            self._writer.fig.clear()
            self.plot_graph(path=sol.path)
            self._writer.grab_frame()
            self._writer.finish()
            self._writer = None
        return sol

    def _run_iteration(self, ax=None):
        _dist, node = heap.heappop(self._pq)
        if node in self._visited.vertex_names:
            return
        self.iteration_count += 1
        # Add node to the visited subgraph along with all of its incoming and outgoing edges to the visited subgraph
        self._visited.add_vertex(self._graph.vertices[node], node)
        edges = self._graph.incident_edges(node)
        for edge in edges:
            if (
                edge.u in self._visited.vertex_names
                and edge.v in self._visited.vertex_names
            ):
                self._visited.add_edge(edge)

        if node == self._graph.source_name:
            self._visited.set_source(self._graph.source_name)

        if self._writer:
            print(f"Iteration {self.iteration_count}, visiting vertex {node}")
            self._writer.fig.clear()
            self.plot_graph()
            self._writer.grab_frame()

        if node == self._graph.target_name:
            return
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

                if new_dist < self._node_dists[neighbor]:
                    self._node_dists[neighbor] = new_dist
                    heap.heappush(self._pq, (new_dist, neighbor))

                if self._writer:
                    self._writer.fig.clear()
                    self.plot_graph(sol.path, edge)
                    self._writer.grab_frame()

    def plot_graph(self, path=None, current_edge=None):
        plt.title("GCS Dijkstra")
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
