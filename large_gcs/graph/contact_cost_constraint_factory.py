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
)
from large_gcs.contact.contact_set import ContactSet


class ContactCostConstraintFactory:
    def __init__(self, vars_pos):
        """We aren't actually using the variables, we are just using them as templates"""
        # Variables for a given vertex/set
        self.vars_pos = vars_pos
        self.vars_all = ContactSet.flatten_set_vars(vars_pos)

        # Create dummy variables for u and v vertices of an edge
        self.u_vars_pos = self.create_vars_from_template(vars_pos, "u")
        self.v_vars_pos = self.create_vars_from_template(vars_pos, "v")

        # Flatten vertex variables
        self.uv_vars_all = np.array(
            [
                ContactSet.flatten_set_vars(self.u_vars_pos),
                ContactSet.flatten_set_vars(self.v_vars_pos),
            ]
        ).flatten()

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
        self.vars_pos has shape (Euclidean/base dim, num positions/pos order per set)
        So to get the path length we need to diff over the second axis.
        """
        exprs = np.diff(self.vars_pos).flatten()
        A = DecomposeLinearExpressions(exprs, self.vars_all)
        b = np.zeros(A.shape[0])
        return L2NormCost(A, b)

    def vertex_cost_position_path_length_squared(self) -> QuadraticCost:
        path_length = np.diff(self.vars_pos).flatten()
        expr = np.dot(path_length, path_length)
        var_map = {var.get_id(): i for i, var in enumerate(self.vars_all)}
        Q, b, c = DecomposeQuadraticPolynomial(Polynomial(expr), var_map)
        return QuadraticCost(Q, b, c)

    ### VERTEX CONSTRAINT CREATION ###

    ### EDGE COST CREATION ###

    def edge_cost_constant(self, constant_cost: float = 1) -> LinearCost:
        """Creates a cost that penalizes each active edge a constant value."""
        # Linear cost of the form: a'x + b, where a is a vector of coefficients and b is a constant.
        a = np.zeros((self.uv_vars_all.size, 1))
        b = constant_cost
        return LinearCost(a, b)

    ### EDGE CONSTRAINT CREATION ###

    def edge_constraint_position_continuity(self) -> List[LinearEqualityConstraint]:
        """Creates a constraint that enforces position continuity between the last position in vertex u to
        the first position in vertex v, given there's an edge from u to v. Since this is an edge constraint,
        the decision variables will be those of both the u and v vertices.
        """
        # Get the last position in u and first position in v
        u_last_pos = self.u_vars_pos[:, :, -1]
        v_first_pos = self.v_vars_pos[:, :, 0]

        exprs_list = (u_last_pos - v_first_pos).reshape(-1, 1)
        # print(f"u_last_pos: {u_last_pos}")
        # print(f"v_first_pos: {v_first_pos}")
        print(f"exprs: {exprs_list}")
        # print(f"uv_vars_all.shape: {self.uv_vars_all.shape}")
        # print(f"uv_vars_all: {self.uv_vars_all}")
        # Linear equality constraint of the form: Ax = b
        constraints = []
        # var_map = {v.get_id(): i for i, v in enumerate(self.uv_vars_all)}
        for exprs in exprs_list:
            print(f"exprs: {exprs}")
            # A = DecomposeLinearExpressions(exprs, self.uv_vars_all)
            # b = np.zeros((A.shape[0],1))
            # print(f"linear A: {A}")
            # print(f"b: {b}")
            A, b = DecomposeAffineExpressions(exprs, self.uv_vars_all)
            print(f"Affine A: {A}")
            print(f"b: {b}")
            constraints.append(LinearEqualityConstraint(A, b))
        # A = DecomposeLinearExpressions(exprs, self.uv_vars_all)
        # b = np.zeros((A.shape[0],1))
        print(f"constraints: {constraints}")
        return constraints
