import gc
import itertools
import logging
import os
import pickle
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import wandb
from large_gcs.algorithms.search_algorithm import (
    AlgVisParams,
    SearchAlgorithm,
    SearchNode,
    profile_method,
)
from large_gcs.graph.graph import Edge, Graph, ShortestPathSolution
from large_gcs.graph.lower_bound_graph import LowerBoundGraph

logger = logging.getLogger(__name__)


class IxG(SearchAlgorithm):
    def __init__(
        self,
        graph: Graph,
        lbg: LowerBoundGraph,
        eps: float = 1,
        vis_params: Optional[AlgVisParams] = None,
    ):
        super().__init__()
        self._graph = graph
        self._target_name = graph.target_name
        self._eps = eps
        self._vis_params = vis_params
        self._counter = itertools.count(start=0, step=1)
        # Stores the search node with the lowest cost to come found so far for each vertex
        self._S: dict[str, SearchNode] = {}
        # Stores the cost to come found so far for each vertex
        self._g: dict[str, float] = defaultdict(lambda: np.inf)
        self._expanded = set()
        # Priority queue
        self._Q = []

        self._lbg = lbg

        # For logging/metrics
        call_structure = {
            "_run_iteration": [
                "_visit_neighbor",
                "_save_metrics",
            ]
        }
        self._alg_metrics.update_method_call_structure(call_structure)
        self._last_plots_save_time = time.time()
        self._step = 0

        start_node = SearchNode(
            priority=0,
            vertex_name=self._graph.source_name,
            edge_path=[],
            vertex_path=[self._graph.source_name],
            sol=None,
        )
        self._g[self._graph.source_name] = 0
        self.push_node_on_Q(start_node)

    def run(self):
        logger.info(f"Running {self.__class__.__name__}")
        sol: Optional[ShortestPathSolution] = None
        start_time = time.time()

        while sol == None and len(self._Q) > 0:
            sol = self._run_iteration()
            self._alg_metrics.time_wall_clock = time.time() - start_time
        if sol is None:
            logger.warning(
                f"{self.__class__.__name__} failed to find a path to the target."
            )
            return
        logger.info(
            f"{self.__class__.__name__} complete! \ncost: {sol.cost}, time: {sol.time}"
            f"\nvertex path: {np.array(sol.vertex_path)}"
        )
        return sol

    @profile_method
    def _run_iteration(self) -> Optional[ShortestPathSolution]:
        # Get the node with the lowest cost to come
        n: SearchNode = self.pop_node_from_Q()
        # import pdb; pdb.set_trace()
        # Only expand the node if it has not been expanded before
        if n.vertex_name in self._expanded:
            return None
        self._expanded.add(n.vertex_name)
        self._alg_metrics.n_vertices_expanded[0] += 1
        edges = self._graph.outgoing_edges(n.vertex_name)
        self._save_metrics(n, edges)
        for edge in edges:
            # Not part of IxG, should replace solve_convex_restriction with something that can handle repeated vertices
            neighbor_in_path = any(
                (self._graph.edges[e].u == edge.v or self._graph.edges[e].v == edge.v)
                for e in n.edge_path
            )
            if neighbor_in_path:
                continue

            # Check early termination condition
            sol = self._visit_neighbor(n, edge)
            if sol is not None:
                return sol

    @profile_method
    def _visit_neighbor(self, n: SearchNode, edge) -> None:
        if edge.v in self._S:
            self._alg_metrics.n_vertices_revisited[0] += 1
        else:
            self._alg_metrics.n_vertices_visited[0] += 1
        n_next = SearchNode.from_parent(child_vertex_name=edge.v, parent=n)
        self._graph.set_target(n_next.vertex_name)
        sol = self._graph.solve_convex_restriction(
            active_edge_keys=n_next.edge_path,
            skip_post_solve=(not n_next.vertex_name == self._target_name),
        )
        if not sol.is_success:
            logger.debug(f"Path not actually feasible")
            # Path invalid, do nothing, don't add to Q
            return
        else:
            logger.debug(f"Path is feasible")

        if sol.cost < self._g[n_next.vertex_name]:
            self._g[n_next.vertex_name] = sol.cost
            n_next.sol = sol
            # Get priority from LBG
            n_next.priority = sol.cost + self._eps * self._lbg.get_cost_to_go(
                n_next.vertex_name
            )
            self.push_node_on_Q(n_next)
            if n_next.vertex_name == self._target_name:
                self._save_metrics(n_next, [], override_save=True)
                return n_next.sol

    @profile_method
    def _solve_convex_restriction(
        self, active_edge_keys: List[str], skip_post_solve: bool = False
    ):
        sol = self._graph.solve_convex_restriction(active_edge_keys, skip_post_solve)
        self._alg_metrics.update_after_gcs_solve(sol.time)
        return sol

    @profile_method
    def _save_metrics(self, n: SearchNode, edges: List[Edge], override_save=False):
        logger.info(
            f"iter: {self._step}\n{self.alg_metrics}\nnow exploring node {n.vertex_name}'s {len(edges)} neighbors ({n.priority})"
        )
        self._step += 1
        current_time = time.time()
        PERIOD = 1200

        if self._vis_params is not None and (
            override_save or self._last_plots_save_time + PERIOD < current_time
        ):
            log_dir = self._vis_params.log_dir
            # Pie chart of method times
            pie_fig = self._alg_metrics.generate_method_time_piechart()
            pie_fig.write_image(os.path.join(log_dir, "method_times_pie_chart.png"))

            # checkpoint_path_str = self.save_checkpoint()

            if wandb.run is not None:
                # Log the Plotly figure and other metrics to wandb
                wandb.log(
                    {
                        "method_times_pie_chart": wandb.Plotly(pie_fig),
                    },
                    step=self._step,
                )
                # wandb.save(checkpoint_path_str)
            self._last_plots_save_time = current_time
        self.log_metrics_to_wandb(n.priority)

    def log_metrics_to_wandb(self, total_estimated_cost: float):
        if wandb.run is not None:  # not self._S
            wandb.log(
                {
                    "total_estimated_cost": total_estimated_cost,
                    "alg_metrics": self.alg_metrics.to_dict(),
                },
                self._step,
            )
