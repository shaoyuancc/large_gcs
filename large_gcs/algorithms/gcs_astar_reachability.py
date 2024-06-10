import gc
import itertools
import logging
import os
import pickle
import time
from collections import defaultdict, deque
from typing import List, Optional

import numpy as np
from tqdm import tqdm

import wandb
from large_gcs.algorithms.search_algorithm import (
    AlgMetrics,
    AlgVisParams,
    SearchAlgorithm,
    SearchNode,
    TieBreak,
    profile_method,
)
from large_gcs.cost_estimators.cost_estimator import CostEstimator
from large_gcs.domination_checkers.domination_checker import DominationChecker
from large_gcs.graph.graph import Edge, Graph, ShortestPathSolution
from large_gcs.graph.incremental_contact_graph import IncrementalContactGraph

logger = logging.getLogger(__name__)
# tracemalloc.start()


class GcsAstarReachability(SearchAlgorithm):
    """
    Note:
    - Use with factored_collision_free cost estimator not yet implemented.
    In particular, this doesn't use a subgraph, but operates directly on the graph.
    """

    def __init__(
        self,
        graph: Graph,
        cost_estimator: CostEstimator,
        domination_checker: DominationChecker,
        tiebreak: TieBreak = TieBreak.FIFO,
        vis_params: Optional[AlgVisParams] = None,
        should_terminate_early: bool = False,
        should_invert_S: bool = False,
        max_len_S_per_vertex: int = 0,  # 0 means no limit
        load_checkpoint_log_dir: Optional[str] = None,
        override_wall_clock_time: Optional[float] = None,
        save_expansion_order: bool = False,
    ):
        if isinstance(graph, IncrementalContactGraph):
            assert (
                graph._should_add_gcs == True
            ), "Required because operating directly on graph instead of subgraph"
        super().__init__()
        self._graph = graph
        self._target = graph.target_name
        self._cost_estimator = cost_estimator
        self._domination_checker = domination_checker
        self._vis_params = vis_params
        self._should_terminate_early = should_terminate_early
        self._should_invert_S = should_invert_S
        self._max_len_S_per_vertex = max_len_S_per_vertex
        self._cost_estimator.set_alg_metrics(self._alg_metrics)
        self._domination_checker.set_alg_metrics(self._alg_metrics)
        self._tiebreak = tiebreak
        if tiebreak == TieBreak.FIFO or tiebreak == TieBreak.FIFO.name:
            self._counter = itertools.count(start=0, step=1)
        elif tiebreak == TieBreak.LIFO or tiebreak == TieBreak.LIFO.name:
            self._counter = itertools.count(start=0, step=-1)

        if should_invert_S:
            self.add_node_to_S = self.add_node_to_S_left
            self.remove_node_from_S = self.remove_node_from_S_left

        self._load_checkpoint_log_dir = load_checkpoint_log_dir
        self._save_expansion_order = save_expansion_order

        # For logging/metrics
        # Expanded set
        self._expanded: set[str] = set()
        call_structure = {
            "_run_iteration": [
                "_visit_neighbor",
                "_generate_neighbors",
                "_save_metrics",
            ],
            "_visit_neighbor": [
                "_is_dominated",
            ],
        }
        self._alg_metrics.update_method_call_structure(call_structure)
        self._last_plots_save_time = time.time()
        self._step = 0

        # Visited dictionary
        self._S: dict[str, deque[SearchNode]] = defaultdict(deque)
        # Priority queue
        self._Q: list[SearchNode] = []

        start_node = SearchNode(
            priority=0,
            vertex_name=self._graph.source_name,
            edge_path=[],
            vertex_path=[self._graph.source_name],
            sol=None,
        )
        self.push_node_on_Q(start_node)

        self._cost_estimator.setup_subgraph(self._graph)

        if load_checkpoint_log_dir is not None:
            self.load_checkpoint(
                load_checkpoint_log_dir,
                override_wall_clock_time=override_wall_clock_time,
            )
            self._load_graph_from_checkpoint_data()

        n_unreachable = gc.collect()
        logger.debug(f"Garbage collected {n_unreachable} unreachable objects")

    def add_node_to_S_left(self, n: SearchNode):
        self._S[n.vertex_name].appendleft(n)

    def add_node_to_S(self, n: SearchNode):
        self._S[n.vertex_name].append(n)

    def remove_node_from_S_left(self, vertex_name: str):
        self._S[vertex_name].pop()

    def remove_node_from_back_of_S(self, vertex_name: str):
        self._S[vertex_name].popleft()

    def run(self) -> ShortestPathSolution:
        """Searches for a shortest path in the given graph."""
        logger.info(f"Running {self.__class__.__name__}")
        if self._load_checkpoint_log_dir is not None:
            start_time = time.time() - self._alg_metrics.time_wall_clock
        else:
            start_time = time.time()
        sol: Optional[ShortestPathSolution] = None
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
        """Runs one iteration of the search algorithm."""
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')
        # for stat in top_stats[:10]:
        #     logger.debug(stat)

        n: SearchNode = self.pop_node_from_Q()

        if self._save_expansion_order:
            self._alg_metrics.expansion_order.append(n.vertex_path)

        # Check termination condition
        if n.vertex_name == self._graph.target_name:
            self._save_metrics(n, [], override_save=True)
            return n.sol

        if n.vertex_name in self._expanded:
            self._alg_metrics.n_vertices_reexpanded[0] += 1
        else:
            self._alg_metrics.n_vertices_expanded[0] += 1
            self._expanded.add(n.vertex_name)
            # Generate neighbors that you are about to explore/visit
            self._generate_neighbors(n.vertex_name)
            # self._graph.generate_neighbors(n.vertex_name)

        edges = self._graph.outgoing_edges(n.vertex_name)

        self._save_metrics(n, edges)

        for edge in edges:
            neighbor_in_path = any(
                (self._graph.edges[e].u == edge.v or self._graph.edges[e].v == edge.v)
                for e in n.edge_path
            )
            if neighbor_in_path:
                continue

            early_terminate_sol = self._visit_neighbor(n, edge)
            if early_terminate_sol is not None:
                return early_terminate_sol

    @profile_method
    def _generate_neighbors(self, vertex_name: str) -> None:
        """Generates neighbors for the given vertex.

        Wrapped to allow for profiling.
        """
        self._graph.generate_neighbors(vertex_name)

    @profile_method
    def _visit_neighbor(
        self, n: SearchNode, edge: Edge
    ) -> Optional[ShortestPathSolution]:
        neighbor = edge.v
        if neighbor in self._S:
            self._alg_metrics.n_vertices_revisited[0] += 1
        else:
            self._alg_metrics.n_vertices_visited[0] += 1

        sol: ShortestPathSolution = self._cost_estimator.estimate_cost_on_graph(
            self._graph,
            edge,
            n.edge_path,
            solve_convex_restriction=True,
        )

        if not sol.is_success:
            logger.debug(f"Path not actually feasible")
            # Path invalid, do nothing, don't add to Q
            return
        else:
            logger.debug(f"Path is feasible")

        n_next = SearchNode.from_parent(child_vertex_name=edge.v, parent=n)
        n_next.sol = sol
        n_next.priority = sol.cost
        logger.debug(
            f"Exploring path (length {len(n_next.vertex_path)}) {n_next.vertex_path}"
        )

        # If coming from source or going to target, do not check if path reaches new samples
        if (
            neighbor != self._target
            # and n.vertex_name != self._graph.source_name
            and self._is_dominated(n_next)
        ):
            # Path does not reach new areas, do not add to Q or S
            logger.debug(f"Not added to Q: Path to is dominated")
            self._alg_metrics._S_pruned_counts[n_next.vertex_name] += 1
            return
        logger.debug(f"Added to Q: Path not dominated")
        if (
            self._max_len_S_per_vertex != 0
            and len(self._S[n_next.vertex_name]) >= self._max_len_S_per_vertex
        ):
            self.remove_node_from_back_of_S(n_next.vertex_name)
        self.add_node_to_S(n_next)
        self.push_node_on_Q(n_next)

        # Early Termination
        if self._should_terminate_early and neighbor == self._target:
            logger.info(f"EARLY TERMINATION: Visited path to target.")
            self._save_metrics(n_next, [], override_save=True)
            return n_next.sol

    @profile_method
    def _is_dominated(self, n: SearchNode) -> bool:
        """Checks if the given node is dominated by any other node in the
        visited set."""

        # Check for trivial domination case
        if n.vertex_name not in self._S:
            return False
        return self._domination_checker.is_dominated(n, self._S[n.vertex_name])

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
            # Histogram of paths per vertex
            # Preparing tracked and pruned counts
            tracked_counts = [len(self._S[v]) for v in self._S]
            pruned_counts = [
                self._alg_metrics._S_pruned_counts[v]
                for v in self._alg_metrics._S_pruned_counts
            ]
            hist_fig = self.alg_metrics.generate_tracked_pruned_paths_histogram(
                tracked_counts, pruned_counts
            )
            # Save the figure to a file as png
            hist_fig.write_image(os.path.join(log_dir, "paths_per_vertex_hist.png"))

            # Pie chart of method times
            pie_fig = self._alg_metrics.generate_method_time_piechart()
            pie_fig.write_image(os.path.join(log_dir, "method_times_pie_chart.png"))

            checkpoint_path_str = self.save_checkpoint()

            if wandb.run is not None:
                # Log the Plotly figure and other metrics to wandb
                wandb.log(
                    {
                        "paths_per_vertex_hist": wandb.Plotly(hist_fig),
                        "method_times_pie_chart": wandb.Plotly(pie_fig),
                    },
                    step=self._step,
                )
                wandb.save(checkpoint_path_str)

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

    @property
    def alg_metrics(self):
        self._alg_metrics.n_S = sum(len(lst) for lst in self._S.values())
        self._alg_metrics.n_S_pruned = sum(self._alg_metrics._S_pruned_counts.values())
        return super().alg_metrics

    def save_checkpoint(self):
        logger.info("Saving checkpoint")
        log_dir = self._vis_params.log_dir
        file_path = os.path.join(log_dir, "checkpoint.pkl")
        checkpoint_data = {
            "Q": self._Q,
            "S": self._S,
            "expanded": self._expanded,
            "counter": next(self._counter) - 1,
            "step": self._step,
            "alg_metrics": self._alg_metrics,
        }
        with open(file_path, "wb") as f:
            pickle.dump(checkpoint_data, f)
        return str(file_path)

    def load_checkpoint(self, checkpoint_log_dir, override_wall_clock_time=None):
        file_path = os.path.join(checkpoint_log_dir, "checkpoint.pkl")
        with open(file_path, "rb") as f:
            checkpoint_data = pickle.load(f)
        self._Q = checkpoint_data["Q"]
        self._S = checkpoint_data["S"]
        self._expanded = checkpoint_data["expanded"]
        if self._tiebreak == TieBreak.FIFO or self._tiebreak == TieBreak.FIFO.name:
            self._counter = itertools.count(start=checkpoint_data["counter"], step=1)
        elif self._tiebreak == TieBreak.LIFO or self._tiebreak == TieBreak.LIFO.name:
            self._counter = itertools.count(start=checkpoint_data["counter"], step=-1)
        self._step = checkpoint_data["step"]
        self._alg_metrics: AlgMetrics = checkpoint_data["alg_metrics"]
        if override_wall_clock_time is not None:
            self._alg_metrics.time_wall_clock = override_wall_clock_time
        self._cost_estimator.set_alg_metrics(self._alg_metrics)
        self._domination_checker.set_alg_metrics(self._alg_metrics)

    def _load_graph_from_checkpoint_data(self):
        # Create a FIFO queue of nodes to expand
        vertices_to_expand = deque()
        vertices_to_expand.append(self._graph.source_name)
        newly_expanded = set()
        total_vertices = len(self._expanded)

        # Use tqdm with leave=False to prevent it from leaving a progress bar after completion
        with tqdm(
            total=total_vertices, desc="Loading graph from checkpoint data", leave=False
        ) as pbar:
            while len(vertices_to_expand) > 0:
                vertex_name = vertices_to_expand.popleft()
                self._graph.generate_neighbors(vertex_name)
                newly_expanded.add(vertex_name)
                edges = self._graph.outgoing_edges(vertex_name)
                for edge in edges:
                    neighbor = edge.v
                    if (
                        (neighbor in self._expanded)
                        and (neighbor not in newly_expanded)
                        and (neighbor not in vertices_to_expand)
                    ):
                        vertices_to_expand.append(neighbor)
                pbar.update(1)

        logger.info(f"Graph loaded with {len(newly_expanded)} vertices expanded.")
