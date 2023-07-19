from collections import defaultdict
import time
import numpy as np
import heapq as heap
import queue
from IPython.display import clear_output
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
from large_gcs.algorithms.search_algorithm import (
    AlgMetrics,
    SearchAlgorithm,
    AlgVisParams,
)
from large_gcs.graph.graph import Edge, Graph


class GcsAstarConvexRestriction(SearchAlgorithm):
    """Convex Restriction version of GCS A*, where in the subroutine, the order of vertices is fixed."""

    def __init__(
        self,
        graph: Graph,
        should_reexplore: bool = False,
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
        self._should_reexplore = should_reexplore
        self._use_convex_relaxation = use_convex_relaxation
        self._shortcut_edge_cost_factory = shortcut_edge_cost_factory
        self._vis_params = vis_params
        self._writer = None
        self._alg_metrics = AlgMetrics()
        self._gcs_solve_times = np.empty((0,))
        self._candidate_sol = None
        self._pq = []
        self._node_dists = defaultdict(lambda: float("inf"))
        self._visited = Graph(self._graph._default_costs_constraints)
        self._visited_vertices = set()
        # Ensures the source is the first node to be visited, even though the heuristic distance is not 0.
        heap.heappush(self._pq, (0, self._graph.source_name, []))

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
        print(
            f"Running {self.__class__.__name__}, should_rexplore: {self._should_reexplore}, use_convex_relaxation: {self._use_convex_relaxation}, shortcut_edge_cost_factory: {self._shortcut_edge_cost_factory.__name__}"
        )
        if animate:
            metadata = dict(title="Convex Restriction GCS A*", artist="Matplotlib")
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
            print(f"Convex Restriction Gcs A* failed to find a solution.")
        else:
            # clear_output(wait=True)
            print(
                f"Convex Restriction Gcs A* complete! \ncost: {sol.cost}, time: {sol.time}\nvertex path: {np.array(sol.vertex_path)}\n{self.alg_metrics}"
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
        heuristic_cost, node, active_edges = heap.heappop(self._pq)
        if not self._should_reexplore and node in self._visited_vertices:
            return
        if node in self._visited_vertices:
            self._alg_metrics.n_vertices_revisited += 1
        else:
            self._alg_metrics.n_vertices_visited += 1

        self._visited_vertices.add(node)
        self._set_visited_vertices_and_edges(active_edges)

        edges = self._graph.outgoing_edges(node)

        if verbose:
            # clear_output(wait=True)
            print(
                f"\n{self.alg_metrics}\nnow exploring node {node}'s {len(edges)} neighbors ({heuristic_cost})"
            )
        if self._writer:
            self._writer.fig.clear()
            self.plot_graph()
            self._writer.grab_frame()

        for edge in edges:
            neighbor_in_path = any(
                (e.u == edge.v or e.v == edge.v) for e in active_edges
            )
            if (
                not self._should_reexplore and edge.v not in self._visited_vertices
            ) or (self._should_reexplore and not neighbor_in_path):
                if edge.v in self._visited_vertices:
                    self._alg_metrics.n_vertices_reexplored += 1
                else:
                    self._alg_metrics.n_vertices_explored += 1
                self._explore_edge(edge, verbose=verbose)

    def _explore_edge(self, edge: Edge, verbose: bool = False):
        neighbor = edge.v
        assert neighbor != self._graph.target_name

        # Add neighbor and edge temporarily to the visited subgraph
        self._visited.add_vertex(self._graph.vertices[neighbor], neighbor)
        # Check if this neighbor actually has an edge to the target
        # If so, add that edge instead of the shortcut
        if (neighbor, self._graph.target_name) in self._graph.edges:
            edge_to_target = self._graph.edges[(neighbor, self._graph.target_name)]
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
            edge_to_target = Edge(
                neighbor, self._graph.target_name, costs=direct_edge_costs
            )
        self._visited.add_edge(edge)
        self._visited.add_edge(edge_to_target)

        sol = self._visited.solve_convex_restriction(self._visited.edges.values())

        self._update_alg_metrics_after_gcs_solve(sol.time)

        self._visited.remove_edge((edge_to_target.u, edge_to_target.v))
        tmp_active_edges = list(self._visited.edges.values()).copy()

        self._visited.remove_vertex(neighbor)

        if sol.is_success:
            new_dist = sol.cost
            if verbose:
                print(
                    f"edge {edge.u} -> {edge.v} is feasible, new dist: {new_dist}, added to pq {new_dist < self._node_dists[neighbor]}"
                )
            if new_dist < self._node_dists[neighbor]:
                self._node_dists[neighbor] = new_dist
                # Remove the edge from the explored node to the target

                heap.heappush(self._pq, (new_dist, neighbor, tmp_active_edges))
                # Check if this neighbor actually has an edge to the target
                if (neighbor, self._graph.target_name) in self._graph.edges:
                    self._candidate_sol = sol

            if self._writer:
                self._writer.fig.clear()
                self.plot_graph(sol.ambient_path, edge)
                self._writer.grab_frame()

        else:
            if verbose:
                print(f"edge {edge.u} -> {edge.v} not actually feasible")

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
        m.vertex_coverage = round(m.n_vertices_visited / self._graph.n_vertices, 4)
        m.n_edges_visited = self._visited.n_edges
        m.edge_coverage = round(m.n_edges_visited / self._graph.n_edges, 4)
        if m.n_gcs_solves > 0:
            m.gcs_solve_time_iter_mean = m.gcs_solve_time_total / m.n_gcs_solves
            m.gcs_solve_time_iter_std = np.std(self._gcs_solve_times)
        if m.n_gcs_solves > 10:
            m.gcs_solve_time_last_10_mean = np.mean(self._gcs_solve_times[-10:])
        return m
