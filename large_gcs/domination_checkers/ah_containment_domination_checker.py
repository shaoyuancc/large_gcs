import logging
from typing import List, Tuple

import numpy as np
import pypolycontain as pp
import scipy
from pydrake.all import (
    GurobiSolver,
    HPolyhedron,
    L1NormCost,
    LinearConstraint,
    LinearCost,
    MathematicalProgram,
    MathematicalProgramResult,
    SolverOptions,
)

from large_gcs.algorithms.search_algorithm import AlgMetrics, SearchNode, profile_method
from large_gcs.domination_checkers.domination_checker import DominationChecker
from large_gcs.geometry.geometry_utils import (
    create_selection_matrix,
    remove_rows_near_zero,
)
from large_gcs.geometry.nullspace_set import AFFINE_SUBSPACE_TOL, NullspaceSet
from large_gcs.geometry.polyhedron import Polyhedron
from large_gcs.graph.graph import Graph

logger = logging.getLogger(__name__)


class AHContainmentDominationChecker(DominationChecker):
    def __init__(
        self,
        graph: Graph,
        containment_condition: int = -1,
        construct_path_from_nullspaces=False,
    ):
        super().__init__(graph=graph)
        self._containment_condition = containment_condition
        self._construct_path_from_nullspaces = construct_path_from_nullspaces

    def set_alg_metrics(self, alg_metrics: AlgMetrics):
        self._alg_metrics = alg_metrics
        call_structure = {
            "_is_dominated": [
                "is_contained_in",
                "_create_path_AH_polytope_from_full_sets",
                "_create_path_AH_polytope_from_nullspace_sets",
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
        logger.debug(
            f"Checking domination of candidate node terminating at vertex {candidate_node.vertex_name}"
            f"\n via path: {candidate_node.vertex_path}"
        )
        AH_n = self._maybe_create_path_AH_polytope(candidate_node)
        # logger.debug(f"{AH_n.P.H.shape}")
        for alt_n in alternate_nodes:
            logger.debug(
                f"Checking if candidate node is dominated by alternate node with path:"
                f"{alt_n.vertex_path}"
            )
            # Might have been added based on the sample so AH Polyhedron might not have been created yet.
            AH_alt = self._maybe_create_path_AH_polytope(alt_n)
            if self.is_contained_in(AH_n, AH_alt):
                return True
        return False

    def _maybe_create_path_AH_polytope(self, node: SearchNode):
        if self._construct_path_from_nullspaces:
            if node.ah_polyhedron_ns is not None:
                return node.ah_polyhedron_ns
            elif node.ah_polyhedron_fs is not None:
                return node.ah_polyhedron_fs
            (
                result,
                reduce_inequalities_succeeded,
            ) = self._create_path_AH_polytope_from_nullspace_sets(node)
            if not reduce_inequalities_succeeded:
                logger.warn(
                    f"Failed to reduce inequalities for _create_path_AH_polytope_from_nullspace_sets so creating from full sets"
                )
                (
                    result,
                    reduce_inequalities_succeeded,
                ) = self._create_path_AH_polytope_from_full_sets(node)
                node.ah_polyhedron_fs = result
            else:
                node.ah_polyhedron_ns = result
        else:
            if node.ah_polyhedron_fs is not None:
                return node.ah_polyhedron_fs
            elif node.ah_polyhedron_ns is not None:
                return node.ah_polyhedron_ns
            (
                result,
                reduce_inequalities_succeeded,
            ) = self._create_path_AH_polytope_from_full_sets(node)
            if not reduce_inequalities_succeeded:
                logger.warn(
                    f"Failed to reduce inequalities for _create_path_AH_polytope_from_full_sets so creating from nullspace sets"
                )
                (
                    result,
                    reduce_inequalities_succeeded,
                ) = self._create_path_AH_polytope_from_nullspace_sets(node)
                node.ah_polyhedron_ns = result
            else:
                node.ah_polyhedron_fs = result
        return result

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
        K_x, k_x, T_x, t_x = self._nullspace_polyhedron_and_transformation_from_AbCdT(
            A_x, b_x, C_x, d_x, T_x
        )
        A_y, b_y, C_y, d_y = Polyhedron.get_separated_inequality_equality_constraints(
            A_y, b_y
        )
        K_y, k_y, T_y, t_y = self._nullspace_polyhedron_and_transformation_from_AbCdT(
            A_y, b_y, C_y, d_y, T_y
        )

        AH_X, AH_Y = self._create_AH_polytopes(K_x, k_x, T_x, t_x, K_y, k_y, T_y, t_y)

        prog = MathematicalProgram()

        # https://github.com/sadraddini/pypolycontain/blob/master/pypolycontain/containment.py#L123
        # -1 for sufficient condition
        # pick `0` for necessary and sufficient encoding (may be too slow) (2019b)
        pp.subset(prog, AH_X, AH_Y, self._containment_condition)

        return self._solve_containment_prog(prog)

    def _nullspace_polyhedron_and_transformation_from_HPoly_and_T(
        self, h_poly: HPolyhedron, T: np.ndarray, t: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        logger.debug(
            f"nullspace_polyhedron_and_transformation_from_HPoly_and_T: Shape of input h_poly: {h_poly.A().shape}"
        )
        nullspace_set, reduce_inequalities_suceeded = NullspaceSet.from_hpolyhedron(
            h_poly, should_reduce_inequalities=True
        )
        T_prime = T @ nullspace_set.V
        if t is None:
            t_prime = T @ nullspace_set.x_0
        else:
            t_prime = T @ nullspace_set.x_0 + t
            # logger.debug(f"t_prime {t_prime.shape}, T {T.shape}, nullspace_set._x_0 {nullspace_set._x_0.shape}, t {t.shape}")
        logger.debug(
            f"nullspace H: {nullspace_set._set.A().shape}, h: {nullspace_set._set.b().shape}, T_prime: {T_prime.shape}, t_prime: {t_prime.shape}"
        )
        return (
            nullspace_set._set.A(),
            nullspace_set._set.b(),
            T_prime,
            t_prime,
            reduce_inequalities_suceeded,
        )  # , nullspace_set._V, nullspace_set._x_0

    def _nullspace_polyhedron_and_transformation_from_AbCdT(self, A, b, C, d, T):
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
    def _create_path_AH_polytope_from_nullspace_sets(self, node: SearchNode):
        logger.debug(f"create_path_AH_polytope_from_nullspace_sets")
        # import pdb
        # pdb.set_trace()
        prog, full_dim = self.get_nullspace_path_mathematical_program(node)
        # logger.debug(f"full_dim: {full_dim}")
        h_poly = HPolyhedron(prog)
        # logger.debug(f"path prog is empty: {h_poly.IsEmpty()}")
        T_H, t_H = self.get_nullspace_H_transformation(node, full_dim=full_dim)
        (
            K,
            k,
            T,
            t,
            reduce_inequalities_succeeded,
        ) = self._nullspace_polyhedron_and_transformation_from_HPoly_and_T(
            h_poly, T_H, t_H
        )
        # logger.debug(f"K: {K.shape}, k: {k.shape}, T: {T.shape}, t: {t.shape}")
        # logger.debug(f"\nK: \n{K}, \nk: \n{k}, \nT: \n{T}, \nt: \n{t}")
        X = pp.H_polytope(K, k)
        return pp.AH_polytope(t, T, X), reduce_inequalities_succeeded

    @profile_method
    def _create_path_AH_polytope_from_full_sets(self, node: SearchNode):
        logger.debug(f"create_path_AH_polytope_from_full_sets")
        # import pdb
        # pdb.set_trace()
        # A, b, C, d = self.get_path_A_b_C_d(node)
        # total_dims = A.shape[1]
        if self.include_cost_epigraph:
            H, h = self.get_epigraph_matrices(node)
            h_poly = HPolyhedron(H, h)
        else:
            prog = self.get_path_constraint_mathematical_program(node)
            h_poly = HPolyhedron(prog)
        # logger.debug(f"full_dim: {h_poly.ambient_dimension()}")
        T_H = self.get_H_transformation(node, h_poly.ambient_dimension())
        # K, k, T, t = self._nullspace_polyhedron_and_transformation_from_AbCdT(A, b, C, d, T_H)
        (
            K,
            k,
            T,
            t,
            reduce_inequalities_succeeded,
        ) = self._nullspace_polyhedron_and_transformation_from_HPoly_and_T(h_poly, T_H)
        # logger.debug(f"K: {K.shape}, k: {k.shape}, T: {T.shape}, t: {t.shape}")
        # logger.debug(f"\nK: \n{K}, \nk: \n{k}, \nT: \n{T}, \nt: \n{t}")
        X = pp.H_polytope(K, k)
        return pp.AH_polytope(t, T, X), reduce_inequalities_succeeded

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
        solver = GurobiSolver()
        result: MathematicalProgramResult = solver.Solve(
            prog, solver_options=solver_options
        )
        # logger.debug(f"Solver name: {result.get_solver_id().name()}")
        return result.is_success()

    def get_nullspace_path_mathematical_program(
        self, node: SearchNode, add_upper_bound=False, cost_upper_bound=500
    ) -> Tuple[MathematicalProgram, int]:
        """Assumes that the path is feasible."""
        # gcs vertices
        vertices = [self._graph.vertices[name].gcs_vertex for name in node.vertex_path]
        ns_sets: List[NullspaceSet] = [
            self._graph.vertices[name].convex_set.nullspace_set
            for name in node.vertex_path
        ]
        edges = [self._graph.edges[edge].gcs_edge for edge in node.edge_path]

        full_v_dim = sum([v.ambient_dimension() for v in vertices])

        prog = MathematicalProgram()
        # Name the vertices by index since cycles are allowed otherwise might get duplicate names.
        ns_vertex_vars = [
            (
                prog.NewContinuousVariables(ns_set.dim, name=f"v{v_idx}_ns_vars")
                if ns_set.dim > 0
                else None
            )
            for v_idx, ns_set in enumerate(ns_sets)
        ]
        ns_dim = sum([ns_set.dim for ns_set in ns_sets])
        for idx, (v, ns_set, lam) in enumerate(zip(vertices, ns_sets, ns_vertex_vars)):
            # Handle the case where the affine subspace is a point
            if lam is None:
                continue

            ns_set.set.AddPointInSetConstraints(prog, lam)

            # Vertex Constraints
            for binding in v.GetConstraints():
                raise NotImplementedError(
                    "Additional vertex constraints not supported yet"
                )
                constraint = binding.evaluator()
                if not isinstance(constraint, LinearConstraint):
                    raise NotImplementedError(
                        f"Only linear constraints are supported for now, {constraint} not supported"
                    )
                lb = constraint.lower_bound() - constraint.GetDenseA() @ ns_set.x_0
                ub = constraint.upper_bound() - constraint.GetDenseA() @ ns_set.x_0
                A = constraint.GetDenseA() @ ns_set.V
                prog.AddLinearConstraint(A=A, lb=lb, ub=ub, vars=lam)

            if self.include_cost_epigraph:
                # Vertex Costs
                for binding in v.GetCosts():
                    cost = binding.evaluator()
                    if isinstance(cost, L1NormCost):
                        A = cost.A()
                        # logger.debug(f"A:\n{A}")
                        l = prog.NewContinuousVariables(
                            A.shape[0], name=f"v{idx}_vertex_l1norm_cost"
                        )
                        prog.AddLinearCost(np.ones(l.shape), l)
                        A_prime = np.hstack((A @ ns_set.V, -np.eye(A.shape[0])))
                        variables = np.hstack((lam, l))
                        b_prime = -A @ ns_set.x_0

                        # logger.debug(f"A_prime {A_prime.shape}, b_prime {b_prime.shape}")
                        # logger.debug(f"\nA_prime\n{A_prime}\nb_prime\n{b_prime}")
                        prog.AddLinearConstraint(
                            A=A_prime,
                            lb=np.full_like(b_prime, -np.inf),
                            ub=b_prime,
                            vars=variables,
                        )
                        A_prime = np.hstack((-A @ ns_set.V, -np.eye(A.shape[0])))
                        b_prime = A @ ns_set.x_0
                        prog.AddLinearConstraint(
                            A=A_prime,
                            lb=np.full_like(b_prime, -np.inf),
                            ub=b_prime,
                            vars=variables,
                        )
                    else:
                        raise NotImplementedError()

        for i, e in enumerate(edges):
            # u, v = ne_nameode.vertex_path[i], node.vertex_path[i + 1]
            u_vars, v_vars = ns_vertex_vars[i], ns_vertex_vars[i + 1]
            u_set, v_set = ns_sets[i], ns_sets[i + 1]
            # Both are points
            if u_vars is None and v_vars is None:
                continue
            # u is a point, v not a point
            elif u_vars is None:
                # Edge Constraints
                for binding in e.GetConstraints():
                    constraint = binding.evaluator()
                    if not isinstance(constraint, LinearConstraint):
                        raise NotImplementedError(
                            f"Only linear constraints are supported for now, {constraint} not supported"
                        )
                    Au, Av = (
                        constraint.GetDenseA()[:, : u_set.x_0.size],
                        constraint.GetDenseA()[:, u_set.x_0.size :],
                    )
                    # logger.debug(f"Adding edge constraint for edge {i}")
                    # logger.debug(f"Au {Au.shape}, Av {Av.shape}, u_set.x_0 {u_set.x_0.shape}, v_set.x_0 {v_set.x_0.shape}")
                    lb = constraint.lower_bound() - Au @ u_set.x_0 - Av @ v_set.x_0
                    ub = constraint.upper_bound() - Au @ u_set.x_0 - Av @ v_set.x_0
                    A = Av @ v_set.V
                    prog.AddLinearConstraint(A=A, lb=lb, ub=ub, vars=v_vars)

                if self.include_cost_epigraph:
                    # Edge Costs
                    for binding in e.GetCosts():
                        cost = binding.evaluator()
                        if isinstance(cost, L1NormCost):
                            raise NotImplementedError()
                        elif isinstance(cost, LinearCost):
                            au, av = np.split(cost.a(), [u_set.x_0.size])
                            a_prime = av @ v_set.V
                            b_prime = au @ u_set.x_0 + av @ v_set.x_0 + cost.b()
                            prog.AddLinearCost(a_prime, b_prime, v_vars)
                        else:
                            raise NotImplementedError()

            # u not a point, v is a point
            elif v_vars is None:
                # Edge Constraints
                for binding in e.GetConstraints():
                    constraint = binding.evaluator()
                    if not isinstance(constraint, LinearConstraint):
                        raise NotImplementedError(
                            f"Only linear constraints are supported for now, {constraint} not supported"
                        )
                    Au, Av = (
                        constraint.GetDenseA()[:, : u_set.x_0.size],
                        constraint.GetDenseA()[:, u_set.x_0.size :],
                    )
                    lb = constraint.lower_bound() - Au @ u_set.x_0 - Av @ v_set.x_0
                    ub = constraint.upper_bound() - Au @ u_set.x_0 - Av @ v_set.x_0
                    A = Au @ u_set.V
                    prog.AddLinearConstraint(A=A, lb=lb, ub=ub, vars=u_vars)

                    if self.include_cost_epigraph:
                        # Edge Costs
                        for binding in e.GetCosts():
                            cost = binding.evaluator()
                            if isinstance(cost, L1NormCost):
                                raise NotImplementedError()
                            elif isinstance(cost, LinearCost):
                                au, av = np.split(cost.a(), [u_set.x_0.size])
                                a_prime = au @ u_set.V
                                b_prime = au @ u_set.x_0 + av @ v_set.x_0 + cost.b()
                                prog.AddLinearCost(a_prime, b_prime, u_vars)
                            else:
                                raise NotImplementedError()

            # Both are not points
            else:
                # Edge Constraints
                for binding in e.GetConstraints():
                    constraint = binding.evaluator()
                    if not isinstance(constraint, LinearConstraint):
                        raise NotImplementedError(
                            f"Only linear constraints are supported for now, {constraint} not supported"
                        )
                    variables = np.concatenate(
                        (ns_vertex_vars[i], ns_vertex_vars[i + 1])
                    )
                    # logger.debug(f"Adding edge constraint for edge {i}")
                    x_0s = np.concatenate((ns_sets[i].x_0, ns_sets[i + 1].x_0))
                    # logger.debug(
                    #     f"ns_sets[{i}].V {ns_sets[i].V.shape}, ns_sets[{i+1}].V {ns_sets[i+1].V.shape}"
                    # )
                    Vs = scipy.linalg.block_diag(ns_sets[i].V, ns_sets[i + 1].V)
                    lb = constraint.lower_bound() - constraint.GetDenseA() @ x_0s
                    ub = constraint.upper_bound() - constraint.GetDenseA() @ x_0s
                    # logger.debug(
                    #     f"constraint.GetDenseA() {constraint.GetDenseA().shape}, Vs {Vs.shape}, lb {lb.shape}, ub {ub.shape}"
                    # )
                    A = constraint.GetDenseA() @ Vs
                    A, lb, ub = remove_rows_near_zero(A, lb, ub, AFFINE_SUBSPACE_TOL)
                    # logger.debug(f"A: {A.shape}")
                    # logger.debug(f"lb: {lb}")
                    # logger.debug(f"ub: {ub}")
                    prog.AddLinearConstraint(A=A, lb=lb, ub=ub, vars=variables)

                if self.include_cost_epigraph:
                    # Edge Costs
                    e_vars = np.hstack((u_vars, v_vars))
                    x_0s = np.concatenate((ns_sets[i].x_0, ns_sets[i + 1].x_0))
                    Vs = scipy.linalg.block_diag(ns_sets[i].V, ns_sets[i + 1].V)
                    for binding in e.GetCosts():
                        cost = binding.evaluator()
                        if isinstance(cost, L1NormCost):
                            raise NotImplementedError()
                        elif isinstance(cost, LinearCost):
                            a_prime = cost.a() @ Vs
                            b_prime = cost.a() @ x_0s + cost.b()
                            prog.AddLinearCost(a_prime, b_prime, e_vars)
                        else:
                            raise NotImplementedError()

        # Add cost epigraph variable
        if self.include_cost_epigraph:
            prog.NewContinuousVariables(1, name="cost")
            N = len(prog.decision_variables())
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
                cost_b += cost.b()
                cost_coeff_row += cost.a() @ S

            if add_upper_bound:
                prog.AddLinearConstraint(
                    A=cost_coeff_row,
                    lb=[-cost_upper_bound],
                    ub=[-cost_b],
                    vars=prog.decision_variables(),
                )
            else:
                prog.AddLinearConstraint(
                    A=cost_coeff_row,
                    lb=[-np.inf],
                    ub=[-cost_b],
                    vars=prog.decision_variables(),
                )

            # Full_v_dim is the hallucinated dimension of the full space path prog
            # N is the dim of the nullspace path prog
            # ns_dim is the dim all all the nullspace vars
            # N-ns_dim = number of l's for the cost and the epigraph variable
            full_v_dim = N - ns_dim + full_v_dim

        return prog, full_v_dim

    def get_path_constraint_mathematical_program(
        self, node: SearchNode
    ) -> MathematicalProgram:
        # gcs vertices
        vertices = [self._graph.vertices[name].gcs_vertex for name in node.vertex_path]
        edges = [self._graph.edges[edge].gcs_edge for edge in node.edge_path]

        prog = MathematicalProgram()
        vertex_vars = [
            prog.NewContinuousVariables(v.ambient_dimension(), name=f"v{v_idx}_vars")
            for v_idx, v in enumerate(vertices)
        ]
        for v, x in zip(vertices, vertex_vars):
            v.set().AddPointInSetConstraints(prog, x)

            # Vertex Constraints
            for binding in v.GetConstraints():
                constraint = binding.evaluator()
                prog.AddConstraint(constraint, x)

        for idx, e in enumerate(edges):
            # Edge Constraints
            for binding in e.GetConstraints():
                constraint = binding.evaluator()
                variables = binding.variables()

                u_idx, v_idx = idx, idx + 1
                variables = np.hstack((vertex_vars[u_idx], vertex_vars[v_idx]))
                prog.AddConstraint(constraint, variables)

        return prog

    def get_path_mathematical_program(self, node: SearchNode) -> MathematicalProgram:
        # gcs vertices
        vertices = [self._graph.vertices[name].gcs_vertex for name in node.vertex_path]
        edges = [self._graph.edges[edge].gcs_edge for edge in node.edge_path]

        prog = MathematicalProgram()

        vertex_vars = [
            prog.NewContinuousVariables(v.ambient_dimension(), name=f"v{v_idx}_vars")
            for v_idx, v in enumerate(vertices)
        ]
        for idx, (v, x) in enumerate(zip(vertices, vertex_vars)):
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
                        A.shape[0], name=f"v{idx}_vertex_l1norm_cost"
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

        for idx, e in enumerate(edges):
            u_idx, v_idx = idx, idx + 1
            variables = np.hstack((vertex_vars[u_idx], vertex_vars[v_idx]))
            # Edge Constraints
            for binding in e.GetConstraints():
                constraint = binding.evaluator()
                prog.AddConstraint(constraint, variables)

            # Edge Costs
            for binding in e.GetCosts():
                cost = binding.evaluator()
                if isinstance(cost, L1NormCost):
                    A = cost.A()
                    t = prog.NewContinuousVariables(
                        A.shape[0], name=f"e{idx}_edge_l1norm_cost"
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

    def get_nullspace_H_transformation(
        self,
        node: SearchNode,
        full_dim: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the transformation matrix that will project the polyhedron that
        defines the whole path down to just the dimensions of the last vertex's
        nullspace.

        Can either include the epigraph (include the cost) or just the
        dimensions of the vertex.
        Note: Cost epigraph variable assumed to be the last decision variable in x.
        """
        # logger.debug(f"AHContainmentDominationChecker.get_nullspace_H_transformation")
        S = self.get_H_transformation(node, full_dim)

        Vs = scipy.linalg.block_diag(
            *[
                self._graph.vertices[name].convex_set.nullspace_set._V
                for name in node.vertex_path
            ]
        )
        x_0s = np.concatenate(
            [
                self._graph.vertices[name].convex_set.nullspace_set._x_0
                for name in node.vertex_path
            ]
        )

        if self.include_cost_epigraph:
            v_dim = len(x_0s)
            n_additional_vars = full_dim - v_dim
            # Assumes the cost variable is the last variable
            Vs = scipy.linalg.block_diag(Vs, np.eye(n_additional_vars))
            x_0s = np.concatenate((x_0s, np.zeros(n_additional_vars)))
        T = S @ Vs
        t = S @ x_0s
        return T, t

    def get_H_transformation(
        self,
        node: SearchNode,
        total_dims: int,
    ):
        """Get the transformation matrix that will project the polyhedron that
        defines the whole path down to just the dimensions of the selected
        vertex.

        Note that the decision vairables are arranged:
        [v1, v2, ..., v_end, ts, ... , ts,  cost]
        where the ts are the slack variables for the l1 norm costs.

        Can either include the epigraph (include the cost) or just the
        dimensions of the vertex.
        Note: Cost epigraph variable assumed to be the last decision variable in x.
        """
        # logger.debug(f"AHContainmentDominationChecker.get_H_transformation")
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
        # logger.debug(f"selected_indices: {selected_indices}")
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
