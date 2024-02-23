from typing import List

import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import (
    DecomposeAffineExpressions,
    Formula,
    FormulaKind,
    HPolyhedron,
    Variables,
    le,
)


def is_on_hyperplane(a, b, x):
    """Returns whether x is on the hyperplane defined by ax = b.
    a and x are vectors with the same dimension, b is a scalar.
    """
    return np.isclose(np.dot(a, x), b)


def counter_clockwise_angle_between(v1, v2):
    """Returns the counter-clockwise angle between two vectors."""
    assert len(v1) == len(v2) == 2, "Vectors must be 2D"
    return np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))


def plot_vector(vec, origin, **kwargs):
    plt.quiver(*origin, *vec, angles="xy", scale_units="xy", scale=1, **kwargs)


def scalar_proj_u_onto_v(u, v):
    u = np.array(u)
    v = np.array(v)
    return np.dot(u, v) / np.linalg.norm(v)


def HPolyhedronAbFromConstraints(
    constraints: List[Formula],
    variables: np.ndarray,
    make_bounded: bool = True,
    remove_constraints_not_in_vars: bool = False,
    BOUND: float = 1000.0,
):
    """
    Construct a polyhedron from a list of constraint formulas.
    Args:
        constraints: array of constraint formulas.
        variables: array of variables.
    """
    if make_bounded:
        ub = np.ones(variables.shape) * BOUND
        upper_limits = le(variables, ub)
        lower_limits = le(-ub, variables)
        limits = np.concatenate((upper_limits, lower_limits))
        constraints = np.append(constraints, limits)

    expressions = []
    # print(f"Constructing polyhedron for set {self.id}")
    # print(f"variables: {variables}")
    for formula in constraints:
        # print(formula)
        kind = formula.get_kind()
        lhs, rhs = formula.Unapply()[1]
        if kind == FormulaKind.Eq:
            # Eq constraint ax = b is
            # implemented as ax ≤ b, -ax <= -b
            expressions.append(lhs - rhs)
            expressions.append(rhs - lhs)
        elif kind == FormulaKind.Geq:
            # lhs >= rhs
            # ==> rhs - lhs ≤ 0
            expressions.append(rhs - lhs)
        elif kind == FormulaKind.Leq:
            # lhs ≤ rhs
            # ==> lhs - rhs ≤ 0
            expressions.append(lhs - rhs)
        else:
            raise NotImplementedError("Type of constraint formula not implemented")

    if remove_constraints_not_in_vars:

        def check_all_vars_are_relevant(exp):
            return all(
                [
                    exp_var
                    in Variables(
                        variables
                    )  # Need to convert this to Variables to check contents
                    for exp_var in exp.GetVariables()
                ]
            )

        expressions = list(filter(check_all_vars_are_relevant, expressions))

    # We now have expr ≤ 0 for all expressions
    # ==> we get Ax - b ≤ 0
    A, b_neg = DecomposeAffineExpressions(expressions, variables)

    # Polyhedrons are of the form: Ax <= b
    b = -b_neg

    return A, b


def HPolyhedronFromConstraints(
    constraints: List[Formula],
    variables: np.ndarray,
    make_bounded: bool = True,
    remove_constraints_not_in_vars: bool = False,
    BOUND: float = 1000.0,
):
    """
    Construct a polyhedron from a list of constraint formulas.
    Args:
        constraints: array of constraint formulas.
        variables: array of variables.
    """

    A, b = HPolyhedronAbFromConstraints(
        constraints,
        variables,
        make_bounded,
        remove_constraints_not_in_vars,
        BOUND,
    )
    polyhedron = HPolyhedron(A, b)

    return polyhedron
