import itertools
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib.animation import FFMpegWriter
from pydrake.all import CommonSolverOption, MathematicalProgram, Solve, SolverOptions

from large_gcs.algorithms.search_algorithm import (
    AlgMetrics,
    AlgVisParams,
    SearchAlgorithm,
    SearchNode,
    TieBreak,
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

    def project(self, graph: Graph, node: SearchNode) -> np.ndarray:
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

        prog = MathematicalProgram()
        results = np.full_like(samples, np.nan)
        for idx, sample in enumerate(samples):
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
                logger.warn(
                    f"Failed to project sample {idx} for vertex {vertex_sampled}, original sample: {sample}"
                )
                # logger.warn(f"Failed to project samples node {n.vertex_name}, vertex_path={n.vertex_path}, edge_path={n.edge_path}")

            else:
                results[idx] = result.GetSolution(sample_vars)
        return unique_rows_with_tolerance_ignore_nan(results, tol=1e-3)


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

        # Visited dictionary
        self._S: dict[str, list[SearchNode]] = defaultdict(list)
        # Priority queue
        self._Q: list[SearchNode] = []
        # Expanded set (just for logging/metrics)
        self._expanded: set[str] = set()
        # Keeps track of samples for each vertex(set) in the graph for which the vertex has been visited but the sample has not been reached.
        self._set_samples: dict[str, SetSamples] = {}

        start_node = SearchNode(
            priority=0,
            vertex_name=self._graph.source_name,
            edge_path=[],
            vertex_path=[self._graph.source_name],
            sol=None,
        )
        self.push_node_on_Q(start_node)

        # Shouldn't need the subgraph, just operate directly on the graph
        # # Initialize the "expanded" subgraph on which
        # # all the convex restrictions will be solved.
        # self._subgraph = Graph(self._graph._default_costs_constraints)
        # # Accounting of the full dimensional vertices in the visited subgraph
        # self._subgraph_fd_vertices = set()

        # Shouldn't need to add the target since that should be handled by _set_subgraph
        # # Add the target to the visited subgraph
        # self._subgraph.add_vertex(
        #     self._graph.vertices[self._graph.target_name], self._graph.target_name
        # )
        # self._subgraph.set_target(self._graph.target_name)
        # # Start with the target node in the visited subgraph
        # self.alg_metrics.n_vertices_expanded[0] = 1
        # self._subgraph_fd_vertices.add(self._graph.target_name)

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
            f"\n{self.alg_metrics}"
        )
        return sol

    def _run_iteration(self) -> Optional[ShortestPathSolution]:
        """
        Runs one iteration of the search algorithm.
        """
        n: SearchNode = self.pop_node_from_Q()

        # Check termination condition
        if n.vertex_name == self._graph.target_name:
            return n.sol

        if n.vertex_name in self._expanded:
            self._alg_metrics.n_vertices_reexpanded[0] += 1
        else:
            self._alg_metrics.n_vertices_expanded[0] += 1
            self._expanded.add(n.vertex_name)

        # Generate neighbors that you are about to explore/visit
        # TODO This might be a problem that we are calling this multiple times on the same vertex
        self._graph.generate_neighbors(n.vertex_name)

        edges = self._graph.outgoing_edges(n.vertex_name)

        logger.info(
            f"\n{self.alg_metrics}\nnow exploring node {n.vertex_name}'s {len(edges)} neighbors ({n.priority})"
        )
        self.log_metrics_to_wandb(n.priority)

        for edge in edges:
            neighbor_in_path = any(
                (self._graph.edges[e].u == edge.v or self._graph.edges[e].v == edge.v)
                for e in n.edge_path
            )
            if not neighbor_in_path:
                self._visit_neighbor(n, edge)

    def _visit_neighbor(self, n: SearchNode, edge: Edge) -> None:
        neighbor = edge.v
        if neighbor in self._S:
            self._alg_metrics.n_vertices_revisited[0] += 1
        else:
            self._alg_metrics.n_vertices_visited[0] += 1
        logger.debug(f"exploring edge {edge.key} via path {n.vertex_path}")
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
            logger.info(
                f"Not added to Q: Path to {n_next.vertex_name} does not reach new samples"
            )
            return
        logger.info(f"Added to Q: Path to {n_next.vertex_name} reaches new samples")
        self._S[neighbor] += [n_next]
        self.push_node_on_Q(n_next)

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

        projected_samples = self._set_samples[n_next.vertex_name].project(
            self._graph, n_next
        )
        reached_new = False
        for idx, sample in enumerate(projected_samples):
            # Create a new vertex for the sample and add it to the graph
            sample_vertex_name = f"{n_next.vertex_name}_sample_{idx}"
            self._graph.add_vertex(
                vertex=Vertex(convex_set=Point(sample)), name=sample_vertex_name
            )
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
                    break
                else:
                    # Clean up edge, but leave the sample vertex
                    self._graph.remove_edge(edge_to_sample.key)
            # If no paths can reach the sample, do not need to check more samples
            reached_new = True
            # Clean up
            if sample_vertex_name in self._graph.vertices:
                self._graph.remove_vertex(sample_vertex_name)
            self._graph.set_target(self._target)
            logger.debug(f"Sample {idx} not reached by any previous path.")
            break

        return reached_new


"""
Actually I think we don't need to operate on this subgraph at all but just the whole graph.
The vertices and edges that are not in the path are going to be cleared by the C++ code,
so we don't need to worry about that.
"""
# def _set_subgraph(self, n: SearchNode) -> None:
#     """
#     Set the subgraph to contain only the vertices and edges in the
#     path of the given search node.
#     """
#     vertices_to_add = set(
#         [self._graph.target_name, self._graph.source_name, n.vertex_name]
#     )
#     for _, v in n.path:
#         vertices_to_add.add(v)

#     # Ignore cfree subgraph sets,
#     # Remove full dimensional sets if they aren't in the path
#     # Add all vertices that aren't already inside
#     for v in self._subgraph_fd_vertices.copy():
#         if v not in vertices_to_add:  # We don't want to have it so remove it
#             self._subgraph.remove_vertex(v)
#             self._subgraph_fd_vertices.remove(v)
#         else:  # We do want to have it but it's already in so don't need to add it
#             vertices_to_add.remove(v)

#     for v in vertices_to_add:
#         self._subgraph.add_vertex(self._graph.vertices[v], v)
#         self._subgraph_fd_vertices.add(v)

#     self._subgraph.set_source(self._graph.source_name)
#     self._subgraph.set_target(self._graph.target_name)

#     # Add edges that aren't already in the visited subgraph.
#     for edge_key in n.path:
#         if edge_key not in self._subgraph.edges:
#             self._subgraph.add_edge(self._graph.edges[edge_key])
