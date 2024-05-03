import numpy as np
import pypolycontain as pp
from pydrake.all import HPolyhedron, L1NormCost, MathematicalProgram, Solve

from large_gcs.algorithms.search_algorithm import SearchNode
from large_gcs.domination_checkers.domination_checker import DominationChecker
from large_gcs.graph.graph import Graph


class AHContainmentDominationChecker(DominationChecker):
    def __init__(self, graph: Graph):
        super().__init__(graph=graph)

    def is_dominated():
        pass

    def is_contained_in(self, A_x, b_x, A_y, b_y, n_dim_proj: int) -> bool:
        X = pp.H_polytope(A_x, b_x)
        Y = pp.H_polytope(A_y, b_y)

        T_x = np.hstack(
            (np.zeros((n_dim_proj, A_x.shape[1] - n_dim_proj)), np.eye(n_dim_proj))
        )
        T_y = np.hstack(
            (np.zeros((n_dim_proj, A_y.shape[1] - n_dim_proj)), np.eye(n_dim_proj))
        )

        AH_X = pp.AH_polytope(np.zeros((n_dim_proj, 1)), T_x, X)
        AH_Y = pp.AH_polytope(np.zeros((n_dim_proj, 1)), T_y, Y)

        prog = MathematicalProgram()

        # https://github.com/sadraddini/pypolycontain/blob/master/pypolycontain/containment.py
        # -1 for sufficient condition
        # pick `0` for necessary and sufficient encoding (may be too slow) (2019b)
        pp.subset(prog, AH_X, AH_Y, -1)
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

    def get_epigraph_matrices(
        self, node: SearchNode, add_upper_bound=True, cost_upper_bound=1e4
    ):
        prog = self.get_path_mathematical_program(node)
        X = HPolyhedron(prog)
        print(f"X.A(): {X.A()}")
        print(f"X.b(): {X.b()}")
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

    @staticmethod
    def find_index(list, el):
        for i in range(len(list)):
            if list[i].get_id() == el.get_id():
                return i
        return -1
