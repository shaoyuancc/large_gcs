import heapq as heap
import itertools
import logging
import time
from collections import defaultdict
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display

from large_gcs.algorithms.search_algorithm import (
    AlgVisParams,
    ReexploreLevel,
    SearchAlgorithm,
    TieBreak,
)
from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.graph import Edge, Graph

logger = logging.getLogger(__name__)


class GcsAstarConvexRestriction(SearchAlgorithm):
    """Convex Restriction version of GCS A*, where in the subroutine, the order
    of vertices is fixed."""

    def __init__(
        self,
        graph: Graph,
        cost_estimator: CostEstimator,
        reexplore_level: ReexploreLevel = ReexploreLevel.NONE,
        tiebreak: TieBreak = TieBreak.FIFO,
        vis_params: AlgVisParams = AlgVisParams(),
    ):
        super().__init__()
        self._graph = graph
        self._cost_estimator = cost_estimator
        self._reexplore_level = (
            ReexploreLevel[reexplore_level]
            if type(reexplore_level) == str
            else reexplore_level
        )
        self._vis_params = vis_params
        self._writer = None
        self._cost_estimator.set_alg_metrics(self._alg_metrics)
        self._candidate_sol = None
        # Priority Q
        self._Q = []
        self._feasible_edges = set()
        # Visited dictionary
        self._node_dists = defaultdict(lambda: float("inf"))
        self._subgraph = Graph(self._graph._default_costs_constraints)
        # Accounting of the full dimensional vertices in the visited subgraph
        self._subgraph_fd_vertices = set()
        # Accounting of all the vertices that have ever been visited for revisit management
        self._expanded = set()
        if tiebreak == TieBreak.FIFO or tiebreak == TieBreak.FIFO.name:
            self._counter = itertools.count(start=0, step=1)
        elif tiebreak == TieBreak.LIFO or tiebreak == TieBreak.LIFO.name:
            self._counter = itertools.count(start=0, step=-1)
        # Ensures the source is the first node to be visited, even though the heuristic distance is not 0.
        heap.heappush(
            self._Q, (0, next(self._counter), self._graph.source_name, [], None)
        )
        # Add the target to the visited subgraph
        self._subgraph.add_vertex(
            self._graph.vertices[self._graph.target_name], self._graph.target_name
        )
        self._subgraph.set_target(self._graph.target_name)
        # Start with the target node in the visited subgraph
        self.alg_metrics.n_vertices_expanded[0] = 1
        self._subgraph_fd_vertices.add(self._graph.target_name)

        self._cost_estimator.setup_subgraph(self._subgraph)

    def run(
        self,
        animate_intermediate: bool = False,
        animate_intermediate_sets: Optional[List[str]] = None,
        final_plot: bool = False,
    ):
        logger.info(
            f"Running {self.__class__.__name__}, reexplore_level: {self._reexplore_level}"
        )
        self._animate_intermediate = animate_intermediate
        self._animate_intermediate_sets = animate_intermediate_sets

        self._start_time = time.time()
        while len(self._Q) > 0:
            # Check for termination condition
            curr = self._Q[0][2]
            if curr == self._graph.target_name:
                sol = self._candidate_sol
                self._graph._post_solve(sol)
                logger.info(
                    f"Gcs A* Convex Restriction complete! \ncost: {sol.cost}, time: {sol.time}\nvertex path: {np.array(sol.vertex_path)}\n{self.alg_metrics}"
                )

                return sol

            self._run_iteration()
            self._alg_metrics.time_wall_clock = time.time() - self._start_time

        logger.warning("Gcs A* Convex Restriction failed to find a path to the target.")
        return None

    def _run_iteration(self):
        estimated_cost, _count, node, active_edges, contact_sol = heap.heappop(self._Q)
        if self._reexplore_level == ReexploreLevel.NONE and node in self._expanded:
            return
        if node in self._expanded:
            self._alg_metrics.n_vertices_reexpanded[0] += 1
        else:
            self._alg_metrics.n_vertices_expanded[0] += 1

        self._set_subgraph_vertices_and_edges(node, active_edges)

        edges = self._graph.outgoing_edges(node)

        logger.info(
            f"\n{self.alg_metrics}\nnow exploring node {node}'s {len(edges)} neighbors ({estimated_cost})"
        )
        self.log_metrics_to_wandb(estimated_cost)

        if (
            self._animate_intermediate
            and contact_sol is not None
            and (
                self._animate_intermediate_sets is None
                or node in self._animate_intermediate_sets
            )
        ):
            self._graph.contact_spp_sol = contact_sol
            anim = self._graph.animate_solution()
            display(HTML(anim.to_html5_video()))

        if self._reexplore_level == ReexploreLevel.NONE:
            for edge in edges:
                if edge.v not in self._expanded or edge.v == self._graph.target_name:
                    self._explore_edge(edge, active_edges)
        else:
            for edge in edges:
                neighbor_in_path = any(
                    (
                        self._graph.edges[e].u == edge.v
                        or self._graph.edges[e].v == edge.v
                    )
                    for e in active_edges
                )
                if not neighbor_in_path:
                    self._explore_edge(edge, active_edges)

    def _explore_edge(self, edge: Edge, active_edges: List[str]):
        neighbor = edge.v
        if neighbor in self._expanded:
            self._alg_metrics.n_vertices_revisited[0] += 1
        else:
            self._alg_metrics.n_vertices_visited[0] += 1
        # logger.info(f"exploring edge {edge.u} -> {edge.v}")
        sol = self._cost_estimator.estimate_cost(
            self._subgraph,
            edge,
            active_edges,
            solve_convex_restriction=True,
        )

        if sol.is_success:
            new_dist = sol.cost
            should_add_to_pq = self._should_add_to_pq(neighbor, new_dist)
            logger.debug(
                f"edge {edge.u} -> {edge.v} is feasible, new dist: {new_dist}, added to pq {should_add_to_pq}"
            )
            if should_add_to_pq:
                new_active_edges = active_edges.copy() + [edge.key]
                # Note this assumes graph is contact graph, should break this dependency...
                # But this dependancy is only necessary for visualizing intermediate solutions
                # Counter serves as tiebreaker for nodes with the same distance, to prevent nodes or edges from being compared
                contact_sol = (
                    self._graph.create_contact_spp_sol(
                        sol.vertex_path, sol.ambient_path, ref_graph=self._subgraph
                    )
                    if isinstance(self._graph, ContactGraph)
                    else None
                )
                heap.heappush(
                    self._Q,
                    (
                        new_dist,
                        next(self._counter),
                        neighbor,
                        new_active_edges,
                        contact_sol,
                    ),
                )

            if new_dist < self._node_dists[neighbor]:
                self._node_dists[neighbor] = new_dist
                # Check if this neighbor actually has an edge to the target
                # Check if this neighbor is actually the target
                if neighbor == self._graph.target_name:
                    if self._candidate_sol is None:
                        self._candidate_sol = sol
                    elif sol.cost < self._candidate_sol.cost:
                        self._candidate_sol = sol

            self._feasible_edges.add((edge.u, edge.v))
        else:
            logger.debug(f"edge {edge.u} -> {edge.v} not actually feasible")

    def _should_add_to_pq(self, neighbor, new_dist):
        if self._reexplore_level == ReexploreLevel.FULL:
            return True
        elif self._reexplore_level == ReexploreLevel.PARTIAL:
            return new_dist < self._node_dists[neighbor]
        elif self._reexplore_level == ReexploreLevel.NONE:
            return neighbor not in self._expanded

    def _set_subgraph_vertices_and_edges(self, vertex_name, edge_keys):
        """Also adds source and target regardless of whether they are in
        edges."""
        if not vertex_name in self._expanded:
            self._graph.generate_neighbors(vertex_name)
            self._expanded.add(vertex_name)
        vertices_to_add = set(
            [self._graph.target_name, self._graph.source_name, vertex_name]
        )
        for e_key in edge_keys:
            vertices_to_add.add(self._graph.edges[e_key].v)

        # Ignore cfree subgraph sets,
        # Remove full dimensional sets if they aren't in the path
        # Add all vertices that aren't already inside
        for v in self._subgraph_fd_vertices.copy():
            if v not in vertices_to_add:  # We don't want to have it so remove it
                self._subgraph.remove_vertex(v)
                self._subgraph_fd_vertices.remove(v)
            else:  # We do want to have it but it's already in so don't need to add it
                vertices_to_add.remove(v)

        for v in vertices_to_add:
            self._subgraph.add_vertex(self._graph.vertices[v], v)
            self._subgraph_fd_vertices.add(v)

        self._subgraph.set_source(self._graph.source_name)
        self._subgraph.set_target(self._graph.target_name)

        # Add edges that aren't already in the visited subgraph.
        for edge_key in edge_keys:
            if edge_key not in self._subgraph.edges:
                self._subgraph.add_edge(self._graph.edges[edge_key])

        # logger.debug(f"visited subgraph edges: {self._visited.edge_keys}")

    def plot_graph(self, path=None, current_edge=None, is_final_path=False):
        plt.title("GCS A* Convex Restriction")
        if self._graph.workspace is not None:
            plt.xlim(self._graph.workspace[0])
            plt.ylim(self._graph.workspace[1])
        plt.gca().set_aspect("equal")
        for vertex_name, vertex in self._graph.vertices.items():
            if current_edge and vertex_name == current_edge.u:
                vertex.convex_set.plot(facecolor=self._vis_params.relaxing_from_color)
            elif current_edge and vertex_name == current_edge.v:
                vertex.convex_set.plot(facecolor=self._vis_params.relaxing_to_color)
            elif vertex_name in self._subgraph.vertex_names:
                vertex.convex_set.plot(facecolor=self._vis_params.visited_vertex_color)
            elif vertex_name in [item[1] for item in self._Q]:
                vertex.convex_set.plot(facecolor=self._vis_params.frontier_color)
            else:
                vertex.convex_set.plot()
        for edge_key in self._graph.edge_keys:
            edge = self._graph.edges[edge_key]
            if current_edge and edge.u == current_edge.u and edge.v == current_edge.v:
                self._graph.plot_edge(
                    edge_key, color=self._vis_params.relaxing_edge_color, zorder=3
                )
            elif edge_key in self._subgraph.edge_keys:
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
