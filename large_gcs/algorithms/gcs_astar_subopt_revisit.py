from collections import defaultdict
import numpy as np
import heapq as heap
import queue
import time
from IPython.display import clear_output
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
from large_gcs.algorithms.search_algorithm import (
    AlgMetrics,
    SearchAlgorithm,
    AlgVisParams,
)
from large_gcs.graph.graph import Edge, Graph


class GcsAstarSubOptRevisit(SearchAlgorithm):
    """Suboptimal version of GCS A*"""

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
        self._infeasible_edge_q = queue.Queue()
        self._feasible_edges = set()
        self._node_dists = defaultdict(lambda: float("inf"))
        self._visited = Graph(self._graph._default_costs_constraints)
        # Ensures the source is the first node to be visited, even though the heuristic distance is not 0.
        heap.heappush(self._pq, (0, self._graph.source_name))

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
        while self._candidate_sol is None:
            if len(self._pq) > 0:
                self._run_iteration(verbose=verbose)
            elif not self._infeasible_edge_q.empty():
                edge = self._infeasible_edge_q.get()
                while edge.v in self._visited.vertex_names:
                    if verbose:
                        print(
                            f"previously unreachable vertex {edge.v} already in visited subgraph"
                        )
                    edge = self._infeasible_edge_q.get()
                if verbose:
                    print(f"exploring previously unreachable vertex {edge.v}")
                self._explore_edge(edge, verbose=verbose)
        self._alg_metrics.time_wall_clock = time.time() - self._start_time

        sol = self._candidate_sol
        # clear_output(wait=True)
        print(
            f"Suboptimal Gcs A* complete! \ncost: {sol.cost} \nvertex path: {np.array(sol.vertex_path)}\n{self.alg_metrics}"
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
        heuristic_cost, node = heap.heappop(self._pq)
        # if node in self._visited.vertex_names and node != self._graph.target_name:
        #     return
        # else:
        self._add_vertex_and_edges_to_visited_except_edges_to_target(node)

        if node == self._graph.source_name:
            self._visited.set_source(self._graph.source_name)

        edges = self._graph.outgoing_edges(node)

        if verbose:
            # clear_output(wait=True)
            print(
                f"\n{self.alg_metrics}\nnow exploring node {node}'s {len(edges)} neighbors ({heuristic_cost})"
            )
            print(f"Current vertices: {self._visited.vertex_names}")
        if self._writer:
            self._writer.fig.clear()
            self.plot_graph()
            self._writer.grab_frame()

        for edge in edges:
            neighbor = edge.v
            if neighbor != self._graph.source_name:
                # if (
                #     neighbor not in self._visited.vertex_names
                #     or neighbor == self._graph.target_name
                # ):
                self._explore_edge(edge, verbose=verbose)

    def _explore_edge(self, edge: Edge, verbose: bool = False):
        neighbor = edge.v
        neighbor_in_visited = neighbor in self._visited.vertices
        edge_in_visited = (edge.u, edge.v) in self._visited.edges

        assert neighbor != self._graph.target_name
        # Add neighbor and edge temporarily to the visited subgraph
        if neighbor in self._visited.vertices:
            self._alg_metrics.n_vertices_explored += 1
        else:
            self._alg_metrics.n_vertices_reexplored += 1
            self._visited.add_vertex(self._graph.vertices[neighbor], neighbor)
        # Check if this neighbor actually has an edge to the target
        # If so, add that edge instead of the shortcut
        if (neighbor, self._graph.target_name) in self._graph.edges:
            neighbor_to_target_edge = self._graph.edges[
                (neighbor, self._graph.target_name)
            ]
        else:
            # Add an edge from the neighbor to the target
            direct_edge_costs = None
            if self._shortcut_edge_cost_factory:
                # Note for now this only works with ContactSet and ContactPointSet because
                # they have the vars attribute, and convex_sets in general do not.
                direct_edge_costs = self._shortcut_edge_cost_factory(
                    self._graph.vertices[neighbor].convex_set.vars,
                    self._graph.vertices[self._graph.target_name].convex_set.vars,
                )
            neighbor_to_target_edge = Edge(
                neighbor, self._graph.target_name, costs=direct_edge_costs
            )
        self._visited.add_edge(neighbor_to_target_edge)
        if not edge_in_visited:
            self._visited.add_edge(edge)
        sol = self._visited.solve(self._use_convex_relaxation)

        self._update_alg_metrics_after_gcs_solve(sol.time)

        if not neighbor_in_visited and neighbor != self._graph.target_name:
            # Remove neighbor and associated edges from the visited subgraph
            self._visited.remove_vertex(neighbor)
        elif edge_in_visited:
            self._visited.remove_edge((neighbor, self._graph.target_name))
        else:
            # Just remove the edge from the neighbor to the target from the visited subgraph
            # because the target node must be kept in the visited subgraph
            self._visited.remove_edge((edge.u, edge.v))
            self._visited.remove_edge((neighbor, self._graph.target_name))

        if sol.is_success:
            new_dist = sol.cost
            if verbose:
                print(
                    f"edge {edge.u} -> {edge.v} is feasible, new dist: {new_dist}, added to pq {new_dist < self._node_dists[neighbor]}"
                )
            if new_dist < self._node_dists[neighbor]:
                self._feasible_edges.add((edge.u, edge.v))
                self._node_dists[neighbor] = new_dist
                heap.heappush(self._pq, (new_dist, neighbor))
                # Check if this neighbor actually has an edge to the target
                if (neighbor, self._graph.target_name) in self._graph.edges:
                    self._candidate_sol = sol

            if self._writer:
                self._writer.fig.clear()
                self.plot_graph(sol.ambient_path, edge)
                self._writer.grab_frame()

        else:
            self._infeasible_edge_q.put(edge)
            if verbose:
                print(f"edge {edge.u} -> {edge.v} not actually feasible")

    def _add_vertex_and_edges_to_visited_except_edges_to_target(self, vertex_name):
        # Add node to the visited subgraph along with all of its incoming and outgoing edges to the visited subgraph
        if vertex_name not in self._visited.vertices:
            self._visited.add_vertex(self._graph.vertices[vertex_name], vertex_name)
            self._alg_metrics.n_vertices_visited += 1
        else:
            self._alg_metrics.n_vertices_revisited += 1
        edges = self._graph.incident_edges(vertex_name)
        for edge in edges:
            if (
                edge.u in self._visited.vertex_names
                and edge.v in self._visited.vertex_names
                and edge.v != self._graph.target_name
                # Experimental, only add edges we've verified are feasible
                and (edge.u, edge.v) in self._feasible_edges
                # Since we are now allowing revisits, don't add edges that are already in the visited subgraph
                and (edge.u, edge.v) not in self._visited.edges
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
