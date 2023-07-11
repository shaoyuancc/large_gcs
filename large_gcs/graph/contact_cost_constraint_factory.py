import numpy as np
from typing import List
from pydrake.all import (
    Variable,
    Polynomial,
    DecomposeAffineExpressions,
    DecomposeLinearExpressions,
    DecomposeQuadraticPolynomial,
    L2NormCost,
    LinearCost,
    QuadraticCost,
    LinearEqualityConstraint,
    LinearConstraint,
    BoundingBoxConstraint,
)
from large_gcs.contact.contact_set import ContactSetDecisionVariables


class ContactCostConstraintFactory:
    def __init__(self, set_vars: ContactSetDecisionVariables):
        """We aren't actually using the variables, we are just using them as templates"""
        # Variables for a given vertex/set
        assert isinstance(set_vars, ContactSetDecisionVariables)
        self.vars = set_vars

        # Create dummy variables for u and v vertices of an edge
        u_vars_all = self.create_vars_from_template(self.vars.all, "u")
        v_vars_all = self.create_vars_from_template(self.vars.all, "v")
        self.u_vars_pos = set_vars.pos_from_all(u_vars_all)
        self.v_vars_pos = set_vars.pos_from_all(v_vars_all)

        # Flatten vertex variables
        self.uv_vars_all = np.concatenate((u_vars_all, v_vars_all))

    @staticmethod
    def create_vars_from_template(
        vars_template: np.ndarray, name_prefix: str
    ) -> np.ndarray:
        """Creates a new set of variables from a template"""
        vars_new = np.empty_like(vars_template)
        for i in range(vars_template.size):
            vars_new.flat[i] = Variable(
                f"{name_prefix}_{vars_template.flat[i].get_name()}",
                type=Variable.Type.CONTINUOUS,
            )
        return vars_new

    ### VERTEX COST CREATION ###

    def vertex_cost_position_path_length(self) -> L2NormCost:
        """Creates a vertex cost that penalizes the length of the path in position space.
        self.vars.pos has shape (Euclidean/base dim, num positions/pos order per set)
        So to get the path length we need to diff over the second axis.
        """
        exprs = np.diff(self.vars.pos).flatten()
        A = DecomposeLinearExpressions(exprs, self.vars.all)
        b = np.zeros(A.shape[0])
        # print(f"vertex_cost_position_path_length A: {A}")
        return L2NormCost(A, b)

    def vertex_cost_position_path_length_squared(self) -> QuadraticCost:
        path_length = np.diff(self.vars.pos).flatten()
        expr = np.dot(path_length, path_length)
        var_map = {var.get_id(): i for i, var in enumerate(self.vars.all)}
        Q, b, c = DecomposeQuadraticPolynomial(Polynomial(expr), var_map)
        return QuadraticCost(Q, b, c)

    def vertex_cost_force_actuation_norm_squared(self) -> QuadraticCost:
        """Creates a vertex cost that penalizes the magnitude of the force actuation squared."""
        expr = np.dot(self.vars.force_act.flatten(), self.vars.force_act.flatten())
        var_map = {var.get_id(): i for i, var in enumerate(self.vars.all)}
        Q, b, c = DecomposeQuadraticPolynomial(Polynomial(expr), var_map)
        return QuadraticCost(Q, b, c)

    ### VERTEX CONSTRAINT CREATION ###

    def vertex_constraint_force_act_limits(
        self, lb: np.ndarray, ub: np.ndarray
    ) -> LinearConstraint:
        """Creates a constraint that limits the magnitude of the force actuation in each dimension."""
        assert self.vars.force_act.size > 0
        raise NotImplementedError

    ### EDGE COST CREATION ###

    def edge_cost_constant(self, constant_cost: float = 1) -> LinearCost:
        """Creates a cost that penalizes each active edge a constant value."""
        # Linear cost of the form: a'x + b, where a is a vector of coefficients and b is a constant.
        a = np.zeros((self.uv_vars_all.size, 1))
        b = constant_cost
        return LinearCost(a, b)

    def edge_costs_position_continuity_norm(
        self, linear_scaling: float = 1
    ) -> L2NormCost:
        # Get the last position in u and first position in v
        u_last_pos = self.u_vars_pos[:, :, -1]
        v_first_pos = self.v_vars_pos[:, :, 0]

        exprs = (u_last_pos - v_first_pos).flatten() * linear_scaling
        # print(f"u_last_pos: {u_last_pos}")
        # print(f"v_first_pos: {v_first_pos}")
        # print(f"edge_costs_position_continuity_norm exprs: {exprs}")
        A = DecomposeLinearExpressions(exprs, self.uv_vars_all)
        b = np.zeros(A.shape[0])
        return L2NormCost(A, b)

    ### EDGE CONSTRAINT CREATION ###

    def edge_constraint_position_continuity(self) -> LinearEqualityConstraint:
        """Creates a constraint that enforces position continuity between the last position in vertex u to
        the first position in vertex v, given there's an edge from u to v. Since this is an edge constraint,
        the decision variables will be those of both the u and v vertices.
        """
        # Get the last position in u and first position in v
        u_last_pos = self.u_vars_pos[:, :, -1]
        v_first_pos = self.v_vars_pos[:, :, 0]

        exprs = (u_last_pos - v_first_pos).flatten()
        # print(f"u_last_pos: {u_last_pos}")
        # print(f"v_first_pos: {v_first_pos}")
        # print(f"uv_vars_all.shape: {self.uv_vars_all.shape}")
        # print(f"uv_vars_all: {self.uv_vars_all}")
        # Linear equality constraint of the form: Ax = b
        A, b = DecomposeAffineExpressions(exprs, self.uv_vars_all)
        # print(f"Affine A: {A}")
        # print(f"b: {b}")
        return LinearEqualityConstraint(A, b)

    def edge_constraint_position_continuity_linearconstraint(self) -> LinearConstraint:
        """Creates a constraint that enforces position continuity between the last position in vertex u to
        the first position in vertex v, given there's an edge from u to v. Since this is an edge constraint,
        the decision variables will be those of both the u and v vertices.
        """
        # Get the last position in u and first position in v
        u_last_pos = self.u_vars_pos[:, :, -1].flatten()
        v_first_pos = self.v_vars_pos[:, :, 0].flatten()

        exprs = u_last_pos - v_first_pos
        # print(f"edge_constraint_position_continuity_linearconstraint exprs: {exprs}")
        A = DecomposeLinearExpressions(exprs, self.uv_vars_all)
        tol = 1e-6
        lb = np.ones((A.shape[0], 1)) * -tol
        ub = np.ones((A.shape[0], 1)) * tol
        # print(f"A: {A}")
        return LinearConstraint(A, lb, ub)
