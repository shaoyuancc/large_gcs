import logging
from typing import List, Tuple

import numpy as np
import pypolycontain as pp
import scipy
from pydrake.all import (
    Constraint,
    HPolyhedron,
    L1NormCost,
    LinearConstraint,
    LinearCost,
    LinearEqualityConstraint,
    MathematicalProgram,
    Solve,
    SolverOptions,
)

from large_gcs.algorithms.search_algorithm import AlgMetrics, SearchNode, profile_method
from large_gcs.domination_checkers.domination_checker import DominationChecker
from large_gcs.geometry.geometry_utils import create_selection_matrix
from large_gcs.geometry.polyhedron import Polyhedron
from large_gcs.graph.graph import Graph

logger = logging.getLogger(__name__)


class AHContainmentDominationChecker(DominationChecker):
    def __init__(self, graph: Graph, containment_condition: int = -1):
        super().__init__(graph=graph)
        self._containment_condition = containment_condition

    def set_alg_metrics(self, alg_metrics: AlgMetrics):
        self._alg_metrics = alg_metrics
        call_structure = {
            "_is_dominated": [
                "is_contained_in",
                "_create_path_AH_polytope",
            ],
            "is_contained_in": [
                "_solve_containment_prog",
            ],
        }
        alg_metrics.update_method_call_structure(call_structure)

    def is_dominated(
        self, candidate_node: SearchNode, alternate_nodes: List[SearchNode]
    ) -> bool:
        """Checks if a candidate path is dominated completely by any one of the
        alternate paths."""
        # logger.debug(
        #     f"Checking domination of candidate node terminating at vertex {candidate_node.vertex_name}"
        #     f"\n via path: {candidate_node.vertex_path}"
        # )
        AH_n = self._create_path_AH_polytope(candidate_node)
        for alt_n in alternate_nodes:
            logger.debug(
                f"Checking if candidate node is dominated by alternate node with path:"
                f"{alt_n.vertex_path}"
            )
            AH_alt = self._create_path_AH_polytope(alt_n)
            if self.is_contained_in(AH_n, AH_alt):
                return True
        return False

    @profile_method
    def is_contained_in(
        self, AH_X: pp.objects.AH_polytope, AH_Y: pp.objects.AH_polytope
    ) -> bool:
        logger.debug(f"Checking containment")

        prog = MathematicalProgram()

        # https://github.com/sadraddini/pypolycontain/blob/master/pypolycontain/containment.py#L123
        # -1 for sufficient condition
        # pick `0` for necessary and sufficient encoding (may be too slow) (2019b)
        pp.subset(prog, AH_X, AH_Y, self._containment_condition)

        return self._solve_containment_prog(prog)

    def is_contained_in_w_AbCd_decomposition(
        self, A_x, b_x, T_x, A_y, b_y, T_y
    ) -> bool:
        logger.debug(f"Checking containment")

        A_x, b_x, C_x, d_x = Polyhedron.get_separated_inequality_equality_constraints(
            A_x, b_x
        )
        K_x, k_x, T_x, t_x = self._nullspace_polyhedron_and_transformation(
            A_x, b_x, C_x, d_x, T_x
        )
        A_y, b_y, C_y, d_y = Polyhedron.get_separated_inequality_equality_constraints(
            A_y, b_y
        )
        K_y, k_y, T_y, t_y = self._nullspace_polyhedron_and_transformation(
            A_y, b_y, C_y, d_y, T_y
        )

        AH_X, AH_Y = self._create_AH_polytopes(K_x, k_x, T_x, t_x, K_y, k_y, T_y, t_y)

        prog = MathematicalProgram()

        # https://github.com/sadraddini/pypolycontain/blob/master/pypolycontain/containment.py#L123
        # -1 for sufficient condition
        # pick `0` for necessary and sufficient encoding (may be too slow) (2019b)
        pp.subset(prog, AH_X, AH_Y, self._containment_condition)

        return self._solve_containment_prog(prog)

    def _nullspace_polyhedron_and_transformation(self, A, b, C, d, T):
        A_invalid = A is None or A.shape[0] == 0
        C_invalid = C is None or C.shape[0] == 0

        if A_invalid and C_invalid:
            raise ValueError("A and C cannot both be empty")
        elif A_invalid:
            raise NotImplementedError(
                "The case where A is empty hasn't been implemented yet"
            )
        elif C_invalid:
            # There are no equality constraints
            return A, b, T, np.zeros((T.shape[0], 1))

        # Compute the basis of the null space of C
        V = scipy.linalg.null_space(C)

        # Compute a point in the null space of C
        x_0, residuals, rank, s = np.linalg.lstsq(C, d, rcond=None)

        A_prime = A @ V
        b_prime = b - A @ x_0

        # Create a boolean mask to identify rows to delete
        delete_mask = np.zeros(len(A_prime), dtype=bool)
        # Detect rows with very small A
        for i, (a1, b1) in enumerate(zip(A_prime, b_prime)):
            if np.allclose(a1, 0, atol=1e-3):
                delete_mask[i] = True
        # Filter out the rows to be deleted
        A_prime = A_prime[~delete_mask]
        b_prime = b_prime[~delete_mask]
        X = HPolyhedron(A_prime, b_prime).ReduceInequalities()
        A_prime, b_prime = X.A(), X.b()

        T_prime = T @ V
        t_prime = T @ x_0

        return A_prime, b_prime, T_prime, t_prime

    @profile_method
    def _create_AH_polytopes(self, A_x, b_x, T_x, t_x, A_y, b_y, T_y, t_y):
        X = pp.H_polytope(A_x, b_x)
        Y = pp.H_polytope(A_y, b_y)

        AH_X = pp.AH_polytope(t_x, T_x, X)
        AH_Y = pp.AH_polytope(t_y, T_y, Y)
        return AH_X, AH_Y

    @profile_method
    def _create_path_AH_polytope(self, node: SearchNode):
        A, b, C, d = self.get_path_A_b_C_d(node)
        total_dims = A.shape[1]
        T_H = self.get_H_transformation(node, total_dims=total_dims)
        K, k, T, t = self._nullspace_polyhedron_and_transformation(A, b, C, d, T_H)
        X = pp.H_polytope(K, k)
        return pp.AH_polytope(t, T, X)

    @profile_method
    def _solve_containment_prog(self, prog: MathematicalProgram):
        logger.debug(f"Solving containment prog")
        solver_options = SolverOptions()
        # solver_options.SetOption(
        #     CommonSolverOption.kPrintFileName, str("mosek_log.txt")
        # )
        # solver_options.SetOption(
        #     CommonSolverOption.kPrintToConsole, 1
        # )

        result = Solve(prog, solver_options=solver_options)
        return result.is_success()

    @profile_method
    def get_path_A_b_C_d(
        self, node: SearchNode
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the A, b, C, d matrices that define the polyhedron that
        represents the path.

        Where Ax <= b are the inequality constraints and Cx = d are the
        equality constraints. A.shape = (m, N) where m is the number of
        constraints and N is the total number of decision variables.
        b.shape = (m,) C.shape = (p, N) where p is the number of
        equality constraints d.shape = (p,)
        """
        # ASSUMPTION: polyhedral sets for the vertices. Linear costs for the vertices and edges.

        # First, collect all the decision variables
        vertices = [self._graph.vertices[name].gcs_vertex for name in node.vertex_path]
        edges = [self._graph.edges[edge].gcs_edge for edge in node.edge_path]

        v_dims = [v.ambient_dimension() for v in vertices]
        current_index = 0
        # Collect the indices of the decision variables for each vertex
        x = []
        for i, dim in enumerate(v_dims):
            x.append(list(range(current_index, current_index + dim)))
            current_index += dim
        # Total number of decision variables
        N = current_index

        if self.include_cost_epigraph:
            # Use a mathematical prog to collate the costs
            prog = MathematicalProgram()
            vertex_vars = [
                prog.NewContinuousVariables(
                    v.ambient_dimension(), name=f"{v_name}_vars"
                )
                for v, v_name in zip(vertices, node.vertex_path)
            ]
            # Vertex Costs:
            for v, v_name, v_vars in zip(vertices, node.vertex_path, vertex_vars):
                for binding in v.GetCosts():
                    cost = binding.evaluator()
                    if isinstance(cost, L1NormCost):
                        A = cost.A()
                        t = prog.NewContinuousVariables(
                            A.shape[0], name=f"{v_name}_vertex_l1norm_cost"
                        )
                        prog.AddLinearCost(np.ones(t.shape), t)
                        prog.AddLinearConstraint(
                            A @ v_vars - t,
                            np.ones(A.shape[0]) * (-np.inf),
                            np.zeros(A.shape[0]),
                        )
                        prog.AddLinearConstraint(
                            -A @ v_vars - t,
                            np.ones(A.shape[0]) * (-np.inf),
                            np.zeros(A.shape[0]),
                        )
                    else:
                        prog.AddCost(cost, v_vars)
            # Edge Costs:
            for e, e_name in zip(edges, node.edge_path):
                edge = self._graph.edges[e_name]
                u_name, v_name = edge.u, edge.v
                u_idx, v_idx = node.vertex_path.index(u_name), node.vertex_path.index(
                    v_name
                )
                for binding in e.GetCosts():
                    cost = binding.evaluator()
                    e_vars = np.concatenate((vertex_vars[u_idx], vertex_vars[v_idx]))
                    if isinstance(cost, L1NormCost):
                        A = cost.A()
                        t = prog.NewContinuousVariables(
                            A.shape[0], name=f"{e_name}_edge_l1norm_cost"
                        )
                        prog.AddLinearCost(np.sum(t))
                        prog.AddLinearConstraint(
                            A @ e_vars - t,
                            np.ones(A.shape[0]) * (-np.inf),
                            np.zeros(A.shape[0]),
                        )
                        prog.AddLinearConstraint(
                            -A @ e_vars - t,
                            np.ones(A.shape[0]) * (-np.inf),
                            np.zeros(A.shape[0]),
                        )
                    else:
                        prog.AddCost(cost, e_vars)

            # Add variable for the total cost
            prog.NewContinuousVariables(1, name="cost")

            N = len(prog.decision_variables())

        # logger.debug(
        #     f"Total number of decision variables {N}, sum of v_dims {sum(v_dims)}"
        # )
        # Collect all the inequality and equality constraints
        ASs, bs, CSs, ds = (
            [np.empty((0, N))],
            [np.empty((0))],
            [np.empty((0, N))],
            [np.empty((0))],
        )
        # A_i @ S_i will give the constraints for all the variables so can be vstacked to big_A
        # (m x n) (n x N) = (m x N)

        def process_constraint(constraint: Constraint, S_i):
            # logger.debug(f"constraint {constraint}")
            # logger.debug(f"lower bound {constraint.lower_bound()}")
            # logger.debug(f"upper bound {constraint.upper_bound()}")
            # Note: It is important that this branch comes first since LinearEqualityConstraint is a subclass of LinearConstraint
            if isinstance(constraint, LinearEqualityConstraint):
                # logger.debug(f"Adding linear equality constraint")
                # Add to the equality constraints
                CSs.append(constraint.GetDenseA() @ S_i)
                # Note for equality constraints, the lower and upper bounds are the same
                ds.append(constraint.lower_bound())
                # logger.debug(f"Added to ds, {ds}")
            elif isinstance(constraint, LinearConstraint):
                # logger.debug(f"Adding linear constraint")
                # Add to the inequality constraints
                if not np.all(constraint.lower_bound() == -np.inf):
                    # -Ax <= -b ==> Ax >= b
                    ASs.append((-constraint.GetDenseA()) @ S_i)
                    bs.append(-constraint.lower_bound())
                if not np.all(constraint.upper_bound() == np.inf):
                    ASs.append(constraint.GetDenseA() @ S_i)
                    bs.append(constraint.upper_bound())
            else:
                raise ValueError(f"Unsupported constraint type {type(constraint)}")

        # Process Vertex Constraints
        for i, v_name in enumerate(node.vertex_path):
            convex_set = self._graph.vertices[v_name].convex_set

            S_i = create_selection_matrix(x[i], N)
            # logger.debug(f"S_i.shape {S_i.shape}")
            # Point In Set Constraints
            if convex_set.A is not None and convex_set.A.shape[0] != 0:
                ASs.append(convex_set.A @ S_i)
                bs.append(convex_set.b)
                # logger.debug(f"Point in set {i} added to bs, {bs}")
            if convex_set.C is not None and convex_set.C.shape[0] != 0:
                # logger.debug(f"C.shape {convex_set.C.shape}")
                CSs.append(convex_set.C @ S_i)
                ds.append(convex_set.d)
                # logger.debug(f"Point in set {i} added to ds, {ds}")

            # Vertex Constraints
            # logger.debug(f"processing vertex constraint vertices[{i}]")
            for binding in vertices[i].GetConstraints():
                process_constraint(binding.evaluator(), S_i)

        # Process Edge Constraints
        for i, e in enumerate(edges):
            # Edges will be two adjacent vertices in the path.
            S_i = create_selection_matrix(x[i] + x[i + 1], N)
            # Edge constraints
            # logger.debug(f"processing edge constraint edge[{i}]")
            for binding in e.GetConstraints():
                process_constraint(binding.evaluator(), S_i)

        if self.include_cost_epigraph:
            # Process the extra constraints introduced by the costs
            for binding in prog.GetAllConstraints():
                con_var_indices = prog.FindDecisionVariableIndices(binding.variables())
                S = create_selection_matrix(con_var_indices, N)
                process_constraint(binding.evaluator(), S)

            # Add the epigraph cost as the final row
            cost_coeff_row = np.zeros((1, N))
            # Assumes the cost variable is the last variable
            cost_coeff_row[0, -1] = -1
            cost_b = 0
            for binding in prog.GetAllCosts():
                cost = binding.evaluator()
                if not isinstance(cost, LinearCost):
                    raise NotImplementedError(
                        f"Only linear costs are supported for now, {cost} not supported"
                    )
                cost_var_indices = prog.FindDecisionVariableIndices(binding.variables())
                S = create_selection_matrix(cost_var_indices, N)
                # Linear cost is of the form a^T x + b
                # Transforming it to cost_coeff_row x <=  - cost_b
                # (1, N) (N, 1) = scalar
                # which then becomes: cost in terms of all other variables + cost_b <= cost_var
                a_T = cost.a().reshape(1, -1)
                cost_b += cost.b()
                cost_coeff_row += a_T @ S

            ASs.append(cost_coeff_row)
            bs.append(np.array([-cost_b]))

        # Stack all the ASs, bs, CSs, ds
        big_A = np.vstack(ASs)
        big_b = np.concatenate(bs)
        big_C = np.vstack(CSs)
        big_d = np.concatenate(ds)

        return big_A, big_b, big_C, big_d

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
            for binding in v.GetCosts():
                cost = binding.evaluator()
                if isinstance(cost, L1NormCost):
                    A = cost.A()
                    t = prog.NewContinuousVariables(
                        A.shape[0], name=f"{v_name}_vertex_l1norm_cost"
                    )
                    prog.AddLinearCost(np.sum(t))
                    prog.AddLinearConstraint(
                        A @ x - t, np.ones(A.shape[0]) * (-np.inf), np.zeros(A.shape[0])
                    )
                    prog.AddLinearConstraint(
                        -A @ x - t,
                        np.ones(A.shape[0]) * (-np.inf),
                        np.zeros(A.shape[0]),
                    )
                else:
                    prog.AddCost(cost, x)

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
                    t = prog.NewContinuousVariables(
                        A.shape[0], name=f"{e_name}_edge_l1norm_cost"
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

    def get_feasibility_matrices_via_prog(self, node: SearchNode):
        logger.debug(f"get_feasibility_matrices_via_prog")
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
        b = 0
        for c in cs:
            c_vars = c.variables()
            c_coeff = c.evaluator().a()
            b += c.evaluator().b()
            for i_c_var, c_var in enumerate(c_vars):
                c_var_ind = self.find_index(vars, c_var)

                c_coeff_vec[c_var_ind] += c_coeff[i_c_var]

        col = np.zeros(X.A().shape[0] + 1)
        col[-1] = -1
        A_x = np.hstack((np.vstack([X.A(), c_coeff_vec]), col.reshape(-1, 1)))
        b_x = np.hstack([X.b(), -b])

        if add_upper_bound:
            # Add upper bound constraint
            A_x = np.vstack([A_x, np.zeros(A_x.shape[1])])
            A_x[-1, -1] = 1
            b_x = np.hstack([b_x, cost_upper_bound])

        return A_x, b_x

    def get_H_transformation(
        self,
        node: SearchNode,
        total_dims: int,
    ):
        """Get the transformation matrix that will project the polyhedron that
        defines the whole path down to just the dimensions of the selected
        vertex.

        Can either include the epigraph (include the cost) or just the
        dimensions of the vertex.
        Note: Cost epigraph variable assumed to be the last decision variable in x.
        """
        # First, collect all the decision variables
        v_dims = [
            self._graph.vertices[name].convex_set.dim for name in node.vertex_path
        ]
        current_index = 0
        # Collect the indices of the decision variables for each vertex
        x = []
        for dim in v_dims:
            x.append(list(range(current_index, current_index + dim)))
            current_index += dim
        selected_indices = x[-1]
        if self.include_cost_epigraph:
            # Assumes the cost variable is the last variable
            selected_indices.append(total_dims - 1)
        return create_selection_matrix(selected_indices, total_dims)

    @staticmethod
    def find_index(list, el):
        for i in range(len(list)):
            if list[i].get_id() == el.get_id():
                return i
        return -1

    @property
    def include_cost_epigraph(self):
        raise NotImplementedError("Subclasses must implement this property")
