import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pydrake.all import MathematicalProgram, Solve, SolverOptions

from large_gcs.algorithms.search_algorithm import AlgMetrics, SearchNode, profile_method
from large_gcs.domination_checkers.domination_checker import DominationChecker
from large_gcs.geometry.geometry_utils import unique_rows_with_tolerance_ignore_nan
from large_gcs.geometry.point import Point
from large_gcs.graph.cost_constraint_factory import (
    create_l2norm_squared_vertex_cost_from_point,
)
from large_gcs.graph.graph import Edge, Graph, ShortestPathSolution, Vertex

logger = logging.getLogger(__name__)


@dataclass
class SetSamples:
    vertex_name: str
    samples: np.ndarray

    @classmethod
    def from_vertex(cls, vertex_name: str, vertex: Vertex, num_samples: int):
        if num_samples == 0:
            samples = np.array([])
        elif isinstance(vertex.convex_set, Point):
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

    def init_graph_for_projection(
        self, graph: Graph, node: SearchNode, alg_metrics: AlgMetrics
    ):
        self._alg_metrics = alg_metrics
        self._proj_graph = Graph()

        # Add vertices along the candidate path to the projection graph
        # Leave out the last vertex, as we need to add the cost specific to the sample to it
        for vertex_name in node.vertex_path[:-1]:
            # Only add constraints, not costs
            ref_vertex = graph.vertices[vertex_name]
            vertex = Vertex(ref_vertex.convex_set, constraints=ref_vertex.constraints)
            self._proj_graph.add_vertex(vertex, vertex_name)

        # Add edges along the candidate path to the projection graph
        # Leave out the last edge, as we didn't add the last vertex
        for edge_key in node.edge_path[:-1]:
            ref_edge = graph.edges[edge_key]
            edge = Edge(
                u=ref_edge.u,
                v=ref_edge.v,
                constraints=ref_edge.constraints,
            )
            self._proj_graph.add_edge(edge)

        self._proj_graph.set_source(graph.source_name)

    @profile_method
    def project_single_gcs(
        self, graph: Graph, node: SearchNode, sample: np.ndarray
    ) -> Optional[np.ndarray]:
        if len(self.samples) == 1:
            # HACK: Work around specifically for when using only a single sample.
            # FIX THIS, see failed_to_project_single_sample_bug.ipynb
            costs = None
        else:
            costs = [create_l2norm_squared_vertex_cost_from_point(sample)]
        # cost = create_l2norm_vertex_cost_from_point(sample)

        ref_vertex = graph.vertices[node.vertex_name]
        vertex = Vertex(
            convex_set=ref_vertex.convex_set,
            constraints=ref_vertex.constraints,
            costs=costs,
        )
        self._proj_graph.add_vertex(vertex, node.vertex_name)

        ref_edge = graph.edges[node.edge_path[-1]]
        edge = Edge(
            u=ref_edge.u,
            v=ref_edge.v,
            constraints=ref_edge.constraints,
            # No costs
        )
        self._proj_graph.add_edge(edge)

        self._proj_graph.set_target(node.vertex_name)

        active_edges = node.edge_path

        sol = self._proj_graph.solve_convex_restriction(
            active_edges, skip_post_solve=False, should_return_result=False
        )

        self._alg_metrics.update_after_gcs_solve(sol.time)

        if not sol.is_success:
            logger.error(
                f"Failed to project sample for vertex {node.vertex_name}"
                f"\nnum total samples for this vertex: {len(self.samples)}"
                f"sample: {sample}"
                f"vertex_path: {node.vertex_path}"
            )
            self._proj_graph.remove_vertex(node.vertex_name)
            return None
            # assert sol.is_success, "Failed to project sample"
        proj_sample = sol.ambient_path[-1]

        # Clean up the projection graph
        self._proj_graph.remove_vertex(node.vertex_name)
        # Edge is automatically removed when vertex is removed

        return proj_sample

    @profile_method
    def project_all_gcs(
        self, graph: Graph, node: SearchNode, alg_metrics: AlgMetrics
    ) -> np.ndarray:
        self.init_graph_for_projection(graph, node, alg_metrics)
        results = np.full_like(self.samples, np.nan)
        n_failed = 0
        for idx, sample in enumerate(self.samples):
            proj_sample = self.project_single_gcs(graph, node, sample)
            if proj_sample is None:
                n_failed += 1
                logger.warning(
                    f"Failed to project sample {idx} for vertex {self.vertex_name}"
                )
            else:
                results[idx] = proj_sample
        if n_failed == len(self.samples):
            logger.error(f"Failed to project any samples for vertex {self.vertex_name}")
        filtered_results = unique_rows_with_tolerance_ignore_nan(results, tol=1e-3)
        return filtered_results

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
            logger.error(
                f"Failed to project sample for vertex {node.vertex_name}"
                f"\nnum total samples for this vertex: {len(self.samples)}"
                f"sample: {sample}"
                f"vertex_path: {node.vertex_path}"
            )
            return None
        return result.GetSolution(sample_vars)

    def project_all(self, graph: Graph, node: SearchNode) -> np.ndarray:
        """Project the samples into the subspace of the last vertex in the
        path, such that the projected samples are reachable via the path."""
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
                logger.warning(
                    f"Failed to project sample {idx} for vertex {vertex_sampled}, original sample: {sample}"
                )
                # logger.warning(f"Failed to project samples node {n.vertex_name}, vertex_path={n.vertex_path}, edge_path={n.edge_path}")
            else:
                results[idx] = result.GetSolution(sample_vars)
        if n_failed == len(samples):
            logger.error(f"Failed to project any samples for vertex {vertex_sampled}")
        filtered_results = unique_rows_with_tolerance_ignore_nan(results, tol=1e-3)
        return filtered_results


class SamplingDominationChecker(DominationChecker):
    def __init__(
        self,
        graph: Graph,
        num_samples_per_vertex: int,
        should_use_candidate_sol: bool = False,
    ):
        super().__init__(graph)

        self._num_samples_per_vertex = num_samples_per_vertex
        self._should_use_candidate_sol = should_use_candidate_sol
        # Keeps track of samples for each vertex(set) in the graph.
        # These samples are not used directly but first projected into the feasible subspace of a particular path.
        self._set_samples: dict[str, SetSamples] = {}

    def set_alg_metrics(self, alg_metrics: AlgMetrics):
        self._alg_metrics = alg_metrics
        call_structure = {
            "_is_dominated": [
                "_maybe_add_set_samples",
                "project_single",
                # "project_single_gcs",
                # "project_all_gcs",
            ],
        }
        alg_metrics.update_method_call_structure(call_structure)

    def is_dominated(
        self, candidate_node: SearchNode, alternate_nodes: list[SearchNode]
    ) -> bool:
        is_dominated = True
        self._maybe_add_set_samples(candidate_node.vertex_name)

        samples = []
        if self._should_use_candidate_sol:
            # The last vertex in the ambient path will be the target,
            # The second last would be the candidate vertex
            raise NotImplementedError(
                "There is a bug that is making the candidate sol sample not feasible"
            )
            samples.append(candidate_node.sol.ambient_path[-2])
            logger.debug(f"Sample from candidate sol: {samples[0]}")

        samples += list(self._set_samples[candidate_node.vertex_name].samples)

        for idx, sample in enumerate(
            self._set_samples[candidate_node.vertex_name].samples
        ):
            # logger.debug(f"Checking sample {idx}")
            # if (not self._should_use_candidate_sol and idx == 0) or (
            #     self._should_use_candidate_sol and idx == 1
            # ):
            #     # Before we project the first sample we need to init the graph
            #     # logger.debug(f"Init graph for projection of samples")
            #     self._set_samples[candidate_node.vertex_name].init_graph_for_projection(
            #         self._graph, candidate_node, self._alg_metrics
            #     )

            if self._should_use_candidate_sol and idx == 0:
                # logger.debug(f"Using candidate sol as sample")
                # Candidate sol does not need to be projected
                proj_sample = sample
            else:
                # logger.debug(f"Projecting sample {idx}")
                proj_sample = self._set_samples[
                    candidate_node.vertex_name
                ].project_single(self._graph, candidate_node, sample)

            if proj_sample is None:
                # If the projection failed assume that the candidate is not feasible, and reject the path
                return True

            # Create a new vertex for the sample and add it to the graph
            sample_vertex_name = f"{candidate_node.vertex_name}_sample_{idx}"

            self._add_sample_to_graph(
                sample=proj_sample,
                sample_vertex_name=sample_vertex_name,
                candidate_node=candidate_node,
            )

            candidate_sol, suceeded = self._compute_candidate_sol(
                candidate_node, sample_vertex_name
            )
            if not suceeded:
                self._graph.remove_vertex(sample_vertex_name)
                continue

            any_single_domination = False
            for alt_i, alt_n in enumerate(alternate_nodes):
                # logger.debug(f"Checking alternate path {alt_i} of {len(alternate_nodes)} for sample {idx}")
                alt_sol = self._solve_conv_res_to_sample(alt_n, sample_vertex_name)
                if self._is_single_dominated(candidate_sol, alt_sol):
                    self._graph.remove_vertex(sample_vertex_name)
                    any_single_domination = True
                    break

            if any_single_domination:
                continue

            # If the candidate path is not dominated by any alternate path, for this sample, do not need to check more samples
            is_dominated = False
            if sample_vertex_name in self._graph.vertices:
                self._graph.remove_vertex(sample_vertex_name)
            logger.debug(f"Sample {idx} reached new/cheaper by candidate path")
            break

        self._graph.set_target(self._target)
        return is_dominated

    def _is_single_dominated(
        self, candidate_sol: ShortestPathSolution, alt_sol: ShortestPathSolution
    ) -> bool:
        raise NotImplementedError

    def _compute_candidate_sol(
        self, candidate_node: SearchNode, sample_vertex_name: str
    ) -> Optional[ShortestPathSolution]:
        raise NotImplementedError

    def _add_sample_to_graph(
        self, sample: np.ndarray, sample_vertex_name: str, candidate_node: SearchNode
    ) -> None:
        # logger.debug(f"_add_sample_to_graph")
        self._graph.add_vertex(
            vertex=Vertex(convex_set=Point(sample)), name=sample_vertex_name
        )

    @profile_method
    def _maybe_add_set_samples(self, vertex_name: str) -> None:
        # Subtract 1 from the number of samples needed if we should use the provided sample is provided
        n_samples_needed = (
            self._num_samples_per_vertex - 1
            if self._should_use_candidate_sol
            else self._num_samples_per_vertex
        )

        if vertex_name not in self._set_samples:
            logger.debug(f"Adding samples for {vertex_name}")
            self._set_samples[vertex_name] = SetSamples.from_vertex(
                vertex_name,
                self._graph.vertices[vertex_name],
                n_samples_needed,
            )

    def _solve_conv_res_to_sample(
        self, node: SearchNode, sample_vertex_name: str
    ) -> ShortestPathSolution:
        # Add edge between the sample and the second last vertex in the path
        e = self._graph.edges[node.edge_path[-1]]
        edge_to_sample = Edge(
            u=e.u,
            v=sample_vertex_name,
            costs=e.costs,
            constraints=e.constraints,
        )
        self._graph.add_edge(edge_to_sample)
        self._graph.set_target(sample_vertex_name)
        active_edges = node.edge_path.copy()
        active_edges[-1] = edge_to_sample.key

        sol = self._graph.solve_convex_restriction(active_edges, skip_post_solve=True)
        self._alg_metrics.update_after_gcs_solve(sol.time)
        # Clean up edge, but leave the sample vertex
        self._graph.remove_edge(edge_to_sample.key)
        return sol

    def plot_set_samples(self, vertex_name: str):
        self._maybe_add_set_samples(vertex_name)
        samples = self._set_samples[vertex_name].samples
        self._graph.plot_points(samples, edgecolor="black")
        self._graph.vertices[vertex_name].convex_set.plot()

    def plot_projected_samples(self, node: SearchNode):
        self._maybe_add_set_samples(node.vertex_name)
        projected_samples = self._set_samples[node.vertex_name].project_all_gcs(
            self._graph, node, AlgMetrics()
        )
        self._graph.plot_points(projected_samples, edgecolor="blue")
