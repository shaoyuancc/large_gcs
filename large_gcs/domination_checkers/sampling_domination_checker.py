from typing import Optional
import numpy as np
import logging
from dataclasses import dataclass
from pydrake.all import CommonSolverOption, MathematicalProgram, Solve, SolverOptions

from large_gcs.algorithms.search_algorithm import AlgMetrics, SearchNode
from large_gcs.domination_checkers.domination_checker import DominationChecker
from large_gcs.geometry.geometry_utils import unique_rows_with_tolerance_ignore_nan
from large_gcs.geometry.point import Point
from large_gcs.graph.graph import Edge, Graph, Vertex
from large_gcs.graph.cost_constraint_factory import (
    create_equality_edge_constraint,
    create_l2norm_squared_vertex_cost_from_point,
    create_l2norm_vertex_cost_from_point,
)

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

    def project_single_gcs(
        self, graph: Graph, node: SearchNode, sample: np.ndarray
    ) -> Optional[np.ndarray]:
        cost = create_l2norm_squared_vertex_cost_from_point(sample)
        ref_vertex = graph.vertices[node.vertex_name]
        vertex = Vertex(
            convex_set=ref_vertex.convex_set,
            constraints=ref_vertex.constraints,
            costs=[cost],
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
            active_edges, skip_post_solve=True
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


class SamplingDominationChecker(DominationChecker):
    def __init__(self, graph: Graph, num_samples_per_vertex: int):
        super().__init__(graph)

        self._num_samples_per_vertex = num_samples_per_vertex

        # Keeps track of samples for each vertex(set) in the graph.
        # These samples are not used directly but first projected into the feasible subspace of a particular path.
        self._set_samples: dict[str, SetSamples] = {}
