from typing import Optional

import numpy as np
import pypolycontain as pp
from pydrake.all import HPolyhedron, L1NormCost, MathematicalProgram, Solve

from large_gcs.algorithms.search_algorithm import AlgMetrics, SearchNode, profile_method
from large_gcs.domination_checkers.domination_checker import DominationChecker
from large_gcs.graph.graph import Graph


class AHContainmentDominationChecker(DominationChecker):
    def __init__(self, graph: Graph, containment_condition: int = -1):
        super().__init__(graph=graph)
        self._containment_condition = containment_condition

    def set_alg_metrics(self, alg_metrics: AlgMetrics):
        self._alg_metrics = alg_metrics
        call_structure = {
            "_is_dominated": [
                "is_contained_in",
                "get_feasibility_matrices",
                "get_epigraph_matrices",
            ],
            "is_contained_in": [
                "_create_AH_polytopes",
                "_solve_containment_prog",
            ],
        }
        alg_metrics.update_method_call_structure(call_structure)

    @profile_method
    def is_contained_in(self, A_x, b_x, T_x, A_y, b_y, T_y) -> bool:
        AH_X, AH_Y = self._create_AH_polytopes(A_x, b_x, T_x, A_y, b_y, T_y)

        prog = MathematicalProgram()

        # https://github.com/sadraddini/pypolycontain/blob/master/pypolycontain/containment.py#L123
        # -1 for sufficient condition
        # pick `0` for necessary and sufficient encoding (may be too slow) (2019b)
        pp.subset(prog, AH_X, AH_Y, self._containment_condition)

        return self._solve_containment_prog(prog)

    @profile_method
    def _create_AH_polytopes(self, A_x, b_x, T_x, A_y, b_y, T_y):
        X = pp.H_polytope(A_x, b_x)
        Y = pp.H_polytope(A_y, b_y)

        AH_X = pp.AH_polytope(np.zeros((T_x.shape[0], 1)), T_x, X)
        AH_Y = pp.AH_polytope(np.zeros((T_y.shape[0], 1)), T_y, Y)
        return AH_X, AH_Y

    @profile_method
    def _solve_containment_prog(self, prog: MathematicalProgram):
        result = Solve(prog)
        return result.is_success()

    def get_path_constraint_mathematical_program(
        self, node: SearchNode
    ) -> MathematicalProgram:
        # gcs vertices
        vertices = [self._graph.vertices[name].gcs_vertex for name in node.vertex_path]
        edges = [self._graph.edges[edge].gcs_edge for edge in node.edge_path]

        prog = MathematicalProgram()
        vertex_vars = [
            prog.NewContinuousVariables(v.ambient_dimension(), name=f"{v_name}_vars")
            for v, v_name in zip(vertices, node.vertex_path)
        ]
        for v, v_name, x in zip(vertices, node.vertex_path, vertex_vars):
            v.set().AddPointInSetConstraints(prog, x)

            # Vertex Constraints
            for binding in v.GetConstraints():
                constraint = binding.evaluator()
                prog.AddConstraint(constraint, x)

        for e, e_name in zip(edges, node.edge_path):
            # Edge Constraints
            for binding in e.GetConstraints():
                constraint = binding.evaluator()
                variables = binding.variables()
                u_name, v_name = (
                    self._graph.edges[e_name].u,
                    self._graph.edges[e_name].v,
                )
                u_idx, v_idx = node.vertex_path.index(u_name), node.vertex_path.index(
                    v_name
                )
                variables[: len(vertex_vars[u_idx])] = vertex_vars[u_idx]
                variables[-len(vertex_vars[v_idx]) :] = vertex_vars[v_idx]
                prog.AddConstraint(constraint, variables)

        return prog

    def get_path_mathematical_program(self, node: SearchNode) -> MathematicalProgram:
        # gcs vertices
        vertices = [self._graph.vertices[name].gcs_vertex for name in node.vertex_path]
        edges = [self._graph.edges[edge].gcs_edge for edge in node.edge_path]

        prog = MathematicalProgram()

        vertex_vars = [
            prog.NewContinuousVariables(v.ambient_dimension(), name=f"{v_name}_vars")
            for v, v_name in zip(vertices, node.vertex_path)
        ]
        for v, v_name, x in zip(vertices, node.vertex_path, vertex_vars):
            v.set().AddPointInSetConstraints(prog, x)

            # Vertex Constraints
            for binding in v.GetConstraints():
                constraint = binding.evaluator()
                prog.AddConstraint(constraint, x)

            # Vertex Costs

        for e, e_name in zip(edges, node.edge_path):
            u_name, v_name = self._graph.edges[e_name].u, self._graph.edges[e_name].v
            u_idx, v_idx = node.vertex_path.index(u_name), node.vertex_path.index(
                v_name
            )
            # Edge Constraints
            for binding in e.GetConstraints():
                constraint = binding.evaluator()
                variables = binding.variables()

                variables[: len(vertex_vars[u_idx])] = vertex_vars[u_idx]
                variables[-len(vertex_vars[v_idx]) :] = vertex_vars[v_idx]
                prog.AddConstraint(constraint, variables)

            # Edge Costs
            for binding in e.GetCosts():
                cost = binding.evaluator()
                variables = binding.variables()
                variables[: len(vertex_vars[u_idx])] = vertex_vars[u_idx]
                variables[-len(vertex_vars[v_idx]) :] = vertex_vars[v_idx]
                if isinstance(cost, L1NormCost):
                    A = cost.A()
                    # For now assume that u and v are of the same dimension
                    t = prog.NewContinuousVariables(
                        A.shape[0], name=f"{e_name}_l1norm_cost"
                    )
                    prog.AddLinearCost(np.sum(t))
                    prog.AddLinearConstraint(
                        A @ variables - t,
                        np.ones(A.shape[0]) * (-np.inf),
                        np.zeros(A.shape[0]),
                    )
                    prog.AddLinearConstraint(
                        -A @ variables - t,
                        np.ones(A.shape[0]) * (-np.inf),
                        np.zeros(A.shape[0]),
                    )
                else:
                    prog.AddCost(cost, variables)

        return prog

    @profile_method
    def get_feasibility_matrices(self, node: SearchNode):
        prog = self.get_path_constraint_mathematical_program(node)
        X = HPolyhedron(prog)
        return X.A(), X.b()

    @profile_method
    def get_epigraph_matrices(
        self, node: SearchNode, add_upper_bound=False, cost_upper_bound=1e4
    ):
        prog = self.get_path_mathematical_program(node)
        X = HPolyhedron(prog)
        vars = list(prog.decision_variables())
        cs = prog.GetAllCosts()
        c_coeff_vec = np.zeros(len(vars))
        for c in cs:
            c_vars = c.variables()
            c_coeff = c.evaluator().a()
            for i_c_var, c_var in enumerate(c_vars):
                c_var_ind = self.find_index(vars, c_var)

                c_coeff_vec[c_var_ind] += c_coeff[i_c_var]

        col = np.zeros(X.A().shape[0] + 1)
        col[-1] = -1
        A_x = np.hstack((np.vstack([X.A(), c_coeff_vec]), col.reshape(-1, 1)))
        b_x = np.hstack([X.b(), 0])

        if add_upper_bound:
            # Add upper bound constraint
            A_x = np.vstack([A_x, np.zeros(A_x.shape[1])])
            A_x[-1, -1] = 1
            b_x = np.hstack([b_x, cost_upper_bound])

        return A_x, b_x

    def get_projection_transformation(
        self,
        node: SearchNode,
        A: np.ndarray,
        include_cost_epigraph: bool,
        vertex_idx_to_project_to: Optional[int] = None,
    ):
        """
        Get the transformation matrix that will project the polyhedron that defines the whole path
        down to just the dimensions of the selected vertex. Can either include the epigraph (include the cost) or just the dimensions of the vertex.

        Args:
            - node(SearchNode) : Defines the path which defines the original matrix that will be projected.
            - A(np.ndarray) : The A matrix of the polyhedron (Ax <= b) that will get transformed.
            - include_cost_epigraph(bool) : Whether to include the cost epigraph in the projection.
            Assumed to be the last decision variable in x.
            - vertex_idx_to_project_to(Optional[int]) : Index of the vertex to project to. If None, will project to the last vertex in the path.
        """
        total_dims = A.shape[1]
        if vertex_idx_to_project_to is None:
            vertex_idx_to_project_to = len(node.vertex_path) - 1

        v_dims = [
            self._graph.vertices[name].convex_set.dim for name in node.vertex_path
        ]
        proj_dims = v_dims[vertex_idx_to_project_to]
        if include_cost_epigraph:
            proj_dims += 1
        matrices_to_stack = []
        cols_count = 0
        for i in range(len(node.vertex_path)):
            if i == vertex_idx_to_project_to:
                M = np.eye(v_dims[i])
                if include_cost_epigraph:
                    M = np.vstack([M, np.zeros((1, v_dims[i]))])
                matrices_to_stack.append(M)
            else:
                matrices_to_stack.append(np.zeros((proj_dims, v_dims[i])))
            cols_count += v_dims[i]
        if cols_count < total_dims:
            M = np.zeros((proj_dims, total_dims - cols_count))
            if include_cost_epigraph:
                M[-1, -1] = 1
            matrices_to_stack.append(M)

        return np.hstack(matrices_to_stack)

    @staticmethod
    def find_index(list, el):
        for i in range(len(list)):
            if list[i].get_id() == el.get_id():
                return i
        return -1
