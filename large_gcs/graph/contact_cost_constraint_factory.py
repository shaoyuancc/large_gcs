import numpy as np
from typing import List
from pydrake.all import (
    Variable,
    DecomposeLinearExpressions,
    L2NormCost,
    LinearCost,
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

    ### VERTEX CONSTRAINT CREATION ###

    ### EDGE COST CREATION ###

    def edge_cost_constant(self, constant_cost: float = 10) -> LinearCost:
        """Creates a cost that penalizes each active edge a constant value."""
        # Linear cost of the form: a'x + b, where a is a vector of coefficients and b is a constant.
        a = np.zeros((self.uv_vars_all.size, 1))
        b = constant_cost
        return LinearCost(a, b)

    ### EDGE CONSTRAINT CREATION ###

    def edge_constraint_position_continuity(self) -> LinearEqualityConstraint:
        """Creates a constraint that enforces position continuity between the last position in vertex u to
        the first position in vertex v, given there's an edge from u to v. Since this is an edge constraint,
        the decision variables will be those of both the u and v vertices.
        """
        # Get the last position in u and first position in v
        u_last_pos = self.u_vars_pos[:, -1]
        v_first_pos = self.v_vars_pos[:, 0]

        exprs = v_first_pos - u_last_pos
        # Linear equality constraint of the form: Ax = b
        A = DecomposeLinearExpressions(exprs, self.uv_vars_all)
        b = np.zeros(A.shape[0])
        return LinearEqualityConstraint(A, b)
