import itertools
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
from IPython.display import HTML, display
from matplotlib.animation import FFMpegWriter
from pydrake.all import CommonSolverOption, MathematicalProgram, Solve, SolverOptions

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
from large_gcs.geometry.convex_set import ConvexSet
from large_gcs.geometry.geometry_utils import unique_rows_with_tolerance_ignore_nan
from large_gcs.geometry.point import Point
from large_gcs.graph.cost_constraint_factory import create_equality_edge_constraint
from large_gcs.graph.graph import Edge, Graph, ShortestPathSolution, Vertex
from large_gcs.graph.incremental_contact_graph import IncrementalContactGraph

logger = logging.getLogger(__name__)


@dataclass
class SetSamples:
    vertex_name: str
    samples: np.ndarray

    @classmethod
    def from_vertex(cls, vertex_name: str, vertex: Vertex, num_samples: int):
        if isinstance(vertex.convex_set, Point):
            # Do not sample from them, just use the point.
            samples = np.array([vertex.convex_set.center])
        else:
            # np.random.seed(0)
            samples = vertex.convex_set.get_samples(num_samples)
            # Round the samples to the nearest 1e-6
            # samples = np.round(samples, 6)
        return cls(
            vertex_name=vertex_name,
            samples=samples,
        )

    def project_single(
        self, graph: Graph, node: SearchNode, sample: np.ndarray
    ) -> np.ndarray:
        vertex_names = node.vertex_path
        active_edges = node.edge_path
        vertex_sampled = self.vertex_name
        # gcs vertices
        vertices = [graph.vertices[name].gcs_vertex for name in vertex_names]
        edges = [graph.edges[edge].gcs_edge for edge in active_edges]

        prog = MathematicalProgram()
        vertex_vars = [
            prog.NewContinuousVariables(v.ambient_dimension(), name=f"{v_name}_vars")
            for v, v_name in zip(vertices, vertex_names)
        ]
        sample_vars = vertex_vars[vertex_names.index(vertex_sampled)]
        for v, v_name, x in zip(vertices, vertex_names, vertex_vars):
            if v_name == vertex_sampled:
                v.set().AddPointInSetConstraints(prog, x)
                # Add the distance to the sample as a cost
                prog.AddCost((x - sample).dot(x - sample))
                # Vertex Constraints
                for binding in v.GetConstraints():
                    constraint = binding.evaluator()
                    prog.AddConstraint(constraint, x)
            else:
                v.set().AddPointInSetConstraints(prog, x)

                # Vertex Constraints
                for binding in v.GetConstraints():
                    constraint = binding.evaluator()
                    prog.AddConstraint(constraint, x)

        for e, e_name in zip(edges, active_edges):
            # Edge Constraints
            for binding in e.GetConstraints():
                constraint = binding.evaluator()
                variables = binding.variables()
                u_name, v_name = graph.edges[e_name].u, graph.edges[e_name].v
                u_idx, v_idx = vertex_names.index(u_name), vertex_names.index(v_name)
                variables[: len(vertex_vars[u_idx])] = vertex_vars[u_idx]
                variables[-len(vertex_vars[v_idx]) :] = vertex_vars[v_idx]
                prog.AddConstraint(constraint, variables)

        solver_options = SolverOptions()
        # solver_options.SetOption(
        #     CommonSolverOption.kPrintFileName, str("mosek_log.txt")
        # )
        # solver_options.SetOption(
        #     CommonSolverOption.kPrintToConsole, 1
        # )

        result = Solve(prog, solver_options=solver_options)
        if not result.is_success():
            return None
        return result.GetSolution(sample_vars)

    def project_all(self, graph: Graph, node: SearchNode) -> np.ndarray:
        """
        Project the samples into the subspace of the last vertex in the path,
        such that the projected samples are reachable via the path.
        """
        assert node.vertex_name == self.vertex_name
        active_edges = node.edge_path
        vertex_sampled = self.vertex_name
        samples = self.samples

        vertex_names = node.vertex_path
        # gcs vertices
        vertices = [graph.vertices[name].gcs_vertex for name in vertex_names]
        edges = [graph.edges[edge].gcs_edge for edge in active_edges]

        results = np.full_like(samples, np.nan)
        n_failed = 0
        for idx, sample in enumerate(samples):
            prog = MathematicalProgram()
            vertex_vars = [
                prog.NewContinuousVariables(
                    v.ambient_dimension(), name=f"{v_name}_vars"
                )
                for v, v_name in zip(vertices, vertex_names)
            ]
            sample_vars = vertex_vars[vertex_names.index(vertex_sampled)]
            for v, v_name, x in zip(vertices, vertex_names, vertex_vars):
                if v_name == vertex_sampled:
                    v.set().AddPointInSetConstraints(prog, x)
                    # Add the distance to the sample as a cost
                    prog.AddCost((x - sample).dot(x - sample))
                    # Vertex Constraints
                    for binding in v.GetConstraints():
                        constraint = binding.evaluator()
                        prog.AddConstraint(constraint, x)
                else:
                    v.set().AddPointInSetConstraints(prog, x)

                    # Vertex Constraints
                    for binding in v.GetConstraints():
                        constraint = binding.evaluator()
                        prog.AddConstraint(constraint, x)

            for e, e_name in zip(edges, active_edges):
                # Edge Constraints
                for binding in e.GetConstraints():
                    constraint = binding.evaluator()
                    variables = binding.variables()
                    u_name, v_name = graph.edges[e_name].u, graph.edges[e_name].v
                    u_idx, v_idx = vertex_names.index(u_name), vertex_names.index(
                        v_name
                    )
                    variables[: len(vertex_vars[u_idx])] = vertex_vars[u_idx]
                    variables[-len(vertex_vars[v_idx]) :] = vertex_vars[v_idx]
                    prog.AddConstraint(constraint, variables)

            solver_options = SolverOptions()
            # solver_options.SetOption(
            #     CommonSolverOption.kPrintFileName, str("mosek_log.txt")
            # )
            # solver_options.SetOption(
            #     CommonSolverOption.kPrintToConsole, 1
            # )

            result = Solve(prog, solver_options=solver_options)
            if not result.is_success():
                n_failed += 1
                logger.warn(
                    f"Failed to project sample {idx} for vertex {vertex_sampled}, original sample: {sample}"
                )
                # logger.warn(f"Failed to project samples node {n.vertex_name}, vertex_path={n.vertex_path}, edge_path={n.edge_path}")
            else:
                results[idx] = result.GetSolution(sample_vars)
        if n_failed == len(samples):
            logger.error(f"Failed to project any samples for vertex {vertex_sampled}")
        filtered_results = unique_rows_with_tolerance_ignore_nan(results, tol=1e-3)
        return filtered_results


class GcsAstarReachability(SearchAlgorithm):
    """
    Reachability based satisficing search on a graph of convex sets using Best-First Search.
    Note:
    - Use with factored_collision_free cost estimator not yet implemented.
    In particular, this doesn't use a subgraph, but operates directly on the graph.
    """

    def __init__(
        self,
        graph: Graph,
        cost_estimator: CostEstimator,
        tiebreak: TieBreak = TieBreak.FIFO,
        vis_params: Optional[AlgVisParams] = None,
        num_samples_per_vertex: int = 100,
        log_dir: Optional[str] = None,
    ):
        if isinstance(graph, IncrementalContactGraph):
            assert (
                graph._should_add_gcs == True
            ), "Required because operating directly on graph instead of subgraph"
        super().__init__()
        self._graph = graph
        self._target = graph.target_name
        self._cost_estimator = cost_estimator
        self._vis_params = vis_params
        self._num_samples_per_vertex = num_samples_per_vertex
        self._cost_estimator.set_alg_metrics(self._alg_metrics)
        if tiebreak == TieBreak.FIFO or tiebreak == TieBreak.FIFO.name:
            self._counter = itertools.count(start=0, step=1)
        elif tiebreak == TieBreak.LIFO or tiebreak == TieBreak.LIFO.name:
            self._counter = itertools.count(start=0, step=-1)
        self._log_dir = log_dir

        # For logging/metrics
        # Expanded set
        self._expanded: set[str] = set()
        call_structure = {
            "_run_iteration": [
                "_visit_neighbor",
                "_generate_neighbors",
                "_save_metrics",
            ],
            "_visit_neighbor": ["_reaches_new"],
            "_reaches_new": ["_project"],
        }
        self._alg_metrics.set_method_call_structure(call_structure)
        self._last_plots_save_time = time.time()
        self._step = 0

        # Visited dictionary
        self._S: dict[str, list[SearchNode]] = defaultdict(list)
        self._S_ignored_counts: dict[str, int] = defaultdict(int)
        # Priority queue
        self._Q: list[SearchNode] = []

        # Keeps track of samples for each vertex(set) in the graph.
        # These samples are not used directly but first projected into the feasible subspace of a particular path.
        self._set_samples: dict[str, SetSamples] = {}

        start_node = SearchNode(
            priority=0,
            vertex_name=self._graph.source_name,
            edge_path=[],
            vertex_path=[self._graph.source_name],
            sol=None,
        )
        self.push_node_on_Q(start_node)

        self._cost_estimator.setup_subgraph(self._graph)

    def run(self) -> ShortestPathSolution:
        """
        Searches for a shortest path in the given graph.
        """
        logger.info(f"Running {self.__class__.__name__}")
        start_time = time.time()
        sol: Optional[ShortestPathSolution] = None
        while sol == None and len(self._Q) > 0:
            sol = self._run_iteration()
            self._alg_metrics.time_wall_clock = time.time() - start_time
        if sol is None:
            logger.warn(
                f"{self.__class__.__name__} failed to find a path to the target."
            )
            return

        self._graph._post_solve(sol)
        logger.info(
            f"{self.__class__.__name__} complete! \ncost: {sol.cost}, time: {sol.time}"
            f"\nvertex path: {np.array(sol.vertex_path)}"
        )
        return sol

    @profile_method
    def _run_iteration(self) -> Optional[ShortestPathSolution]:
        """
        Runs one iteration of the search algorithm.
        """
        n: SearchNode = self.pop_node_from_Q()

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
            if not neighbor_in_path:
                self._visit_neighbor(n, edge)

    @profile_method
    def _generate_neighbors(self, vertex_name: str) -> None:
        """
        Generates neighbors for the given vertex.
        Wrapped to allow for profiling.
        """
        self._graph.generate_neighbors(vertex_name)

    @profile_method
    def _visit_neighbor(self, n: SearchNode, edge: Edge) -> None:
        neighbor = edge.v
        if neighbor in self._S:
            self._alg_metrics.n_vertices_revisited[0] += 1
        else:
            self._alg_metrics.n_vertices_visited[0] += 1
        # logger.debug(f"exploring edge {edge.key} via path {n.vertex_path}")
        sol: ShortestPathSolution = self._cost_estimator.estimate_cost_on_graph(
            self._graph,
            edge,
            n.edge_path,
            solve_convex_restriction=True,
        )

        if not sol.is_success:
            logger.debug(f"edge {n.vertex_name} -> {edge.v} not actually feasible")
            # Path invalid, do nothing, don't add to Q
            return
        else:
            logger.debug(f"edge {n.vertex_name} -> {edge.v} is feasible")

        n_next = SearchNode.from_parent(child_vertex_name=edge.v, parent=n)
        n_next.sol = sol
        n_next.priority = sol.cost

        # If coming from source or going to target, do not check if path reaches new samples
        if (
            neighbor != self._target
            # and n.vertex_name != self._graph.source_name
            and not self._reaches_new(n_next)
        ):
            # Path does not reach new areas, do not add to Q or S
            logger.debug(
                f"Not added to Q: Path to {n_next.vertex_name} does not reach new samples"
            )
            self._S_ignored_counts[n_next.vertex_name] += 1
            return
        logger.debug(f"Added to Q: Path to {n_next.vertex_name} reaches new samples")
        self._S[neighbor] += [n_next]
        self.push_node_on_Q(n_next)

    @profile_method
    def _reaches_new(self, n_next: SearchNode) -> bool:
        """
        Checks samples to see if this path reaches new previously unreached samples.
        Assumes that this path is feasible.
        """
        # If the vertex has not been visited before and the path is feasible, it definitely reaches new
        if n_next.vertex_name not in self._S:
            return True

        if n_next.vertex_name not in self._set_samples:
            logger.debug(f"Adding samples for {n_next.vertex_name}")
            self._set_samples[n_next.vertex_name] = SetSamples.from_vertex(
                n_next.vertex_name,
                self._graph.vertices[n_next.vertex_name],
                self._num_samples_per_vertex,
            )

        reached_new = False
        projected_samples = set()
        for idx, sample in enumerate(self._set_samples[n_next.vertex_name].samples):
            sample = self._project(n_next, sample)

            if sample is None:
                logger.warn(
                    f"Failed to project sample {idx} for vertex {n_next.vertex_name}"
                )
                continue
            else:
                rounded_sample = np.round(sample, 6)
                if tuple(rounded_sample) in projected_samples:
                    continue
                projected_samples.add(tuple(rounded_sample))
            # Create a new vertex for the sample and add it to the graph
            sample_vertex_name = f"{n_next.vertex_name}_sample_{idx}"
            self._graph.add_vertex(
                vertex=Vertex(convex_set=Point(sample)), name=sample_vertex_name
            )
            go_to_next_sample = False
            for alt_n in self._S[n_next.vertex_name]:
                # Add edge between the sample and the second last vertex in the path
                e = self._graph.edges[alt_n.edge_path[-1]]
                edge_to_sample = Edge(
                    u=e.u,
                    v=sample_vertex_name,
                    costs=e.costs,
                    constraints=e.constraints,
                )
                self._graph.add_edge(edge_to_sample)
                # Check whether sample can be reached via the path
                self._graph.set_target(sample_vertex_name)
                active_edges = alt_n.edge_path.copy()
                active_edges[-1] = edge_to_sample.key

                sol = self._graph.solve_convex_restriction(
                    active_edges, skip_post_solve=True
                )
                self._alg_metrics.update_after_gcs_solve(sol.time)
                if sol.is_success:
                    # Clean up current sample
                    self._graph.remove_vertex(sample_vertex_name)
                    # Move on to the next sample, don't need to check other paths
                    logger.debug(f"Sample {idx} reached by path {alt_n.vertex_path}")
                    go_to_next_sample = True
                    break
                else:
                    # Clean up edge, but leave the sample vertex
                    self._graph.remove_edge(edge_to_sample.key)
            if go_to_next_sample:
                continue
            # If no paths can reach the sample, do not need to check more samples
            reached_new = True
            # Clean up
            if sample_vertex_name in self._graph.vertices:
                self._graph.remove_vertex(sample_vertex_name)

            logger.debug(f"Sample {idx} not reached by any previous path.")
            break
        self._graph.set_target(self._target)
        return reached_new

    @profile_method
    def _project(self, n_next: SearchNode, sample: np.ndarray) -> np.ndarray:
        sample = self._set_samples[n_next.vertex_name].project_single(
            self._graph, n_next, sample
        )
        return sample

    @profile_method
    def _save_metrics(self, n: SearchNode, edges: List[Edge], override_save=False):
        logger.info(
            f"\n{self.alg_metrics}\nnow exploring node {n.vertex_name}'s {len(edges)} neighbors ({n.priority})"
        )
        self._step += 1
        current_time = time.time()
        PERIOD = 300
        if self._log_dir is not None and (
            override_save or self._last_plots_save_time + PERIOD < current_time
        ):
            # Histogram of paths per vertex
            # Preparing tracked and ignored counts
            tracked_counts = [len(self._S[v]) for v in self._S]
            ignored_counts = [self._S_ignored_counts[v] for v in self._S_ignored_counts]
            hist_fig = self.alg_metrics.generate_tracked_ignored_paths_histogram(
                tracked_counts, ignored_counts
            )
            # Save the figure to a file as png
            hist_fig.write_image(
                os.path.join(self._log_dir, "paths_per_vertex_hist.png")
            )

            # Pie chart of method times
            pie_fig = self._alg_metrics.generate_method_time_piechart()
            pie_fig.write_image(
                os.path.join(self._log_dir, "method_times_pie_chart.png")
            )

            if wandb.run is not None:
                # Log the Plotly figure and other metrics to wandb
                wandb.log(
                    {
                        "paths_per_vertex_hist": wandb.Plotly(hist_fig),
                        "method_times_pie_chart": wandb.Plotly(pie_fig),
                    },
                    step=self._step,
                )

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
        return super().alg_metrics
