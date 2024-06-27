from typing import List

import numpy as np
from pydrake.all import (
    BoundingBoxConstraint,
    Cost,
    DecomposeAffineExpressions,
    DecomposeLinearExpressions,
    L1NormCost,
    L2NormCost,
    LinearConstraint,
    LinearCost,
    LinearEqualityConstraint,
    Variable,
)

from large_gcs.contact.contact_set_decision_variables import ContactSetDecisionVariables


def create_vars_from_template(
    vars_template: np.ndarray, name_prefix: str
) -> np.ndarray:
    """Creates a new set of variables from a template.

    We use this because the names of the variables would be otherwise be
    the same for u and v.
    """
    vars_new = np.empty_like(vars_template)
    for i in range(vars_template.size):
        vars_new.flat[i] = Variable(
            f"{name_prefix}_{vars_template.flat[i].get_name()}",
            type=Variable.Type.CONTINUOUS,
        )
    return vars_new


def create_l1norm_cost(u_last_pos, v_first_pos, uv_vars_all, scaling=1):
    exprs = (u_last_pos - v_first_pos).flatten() * scaling
    A = DecomposeLinearExpressions(exprs, uv_vars_all)
    b = np.zeros(A.shape[0])
    return L1NormCost(A, b)


def create_l2norm_cost(u_last_pos, v_first_pos, uv_vars_all, scaling=1):
    exprs = (u_last_pos - v_first_pos).flatten() * scaling
    A = DecomposeLinearExpressions(exprs, uv_vars_all)
    b = np.zeros(A.shape[0])
    return L2NormCost(A, b)


def create_scaled_l1norm_position_continuity_costs(
    u_vars: ContactSetDecisionVariables,
    v_vars: ContactSetDecisionVariables,
    scaling_eps: float,
):
    u_vars_all = create_vars_from_template(u_vars.all, "u")
    v_vars_all = create_vars_from_template(v_vars.all, "v")
    uv_vars_all = np.concatenate((u_vars_all, v_vars_all))

    costs = [
        create_l1norm_cost(
            u_vars.obj_last_pos_from_all(u_vars_all),
            v_vars.obj_first_pos_from_all(v_vars_all),
            uv_vars_all,
            scaling=1 * scaling_eps,
        ),
        create_l1norm_cost(
            u_vars.rob_last_pos_from_all(u_vars_all),
            v_vars.rob_first_pos_from_all(v_vars_all),
            uv_vars_all,
            scaling=0.2 * scaling_eps,
        ),
    ]
    return costs


def create_scaled_l2norm_position_continuity_costs(
    u_vars: ContactSetDecisionVariables,
    v_vars: ContactSetDecisionVariables,
    scaling_eps: float,
):
    u_vars_all = create_vars_from_template(u_vars.all, "u")
    v_vars_all = create_vars_from_template(v_vars.all, "v")
    uv_vars_all = np.concatenate((u_vars_all, v_vars_all))

    costs = [
        create_l2norm_cost(
            u_vars.obj_last_pos_from_all(u_vars_all),
            v_vars.obj_first_pos_from_all(v_vars_all),
            uv_vars_all,
            scaling=1 * scaling_eps,
        ),
        create_l2norm_cost(
            u_vars.rob_last_pos_from_all(u_vars_all),
            v_vars.rob_first_pos_from_all(v_vars_all),
            uv_vars_all,
            scaling=0.2 * scaling_eps,
        ),
    ]
    return costs


def contact_shortcut_edge_l1norm_cost_factory_obj_weighted(
    u_vars: ContactSetDecisionVariables,
    v_vars: ContactSetDecisionVariables,
    add_const_cost: bool = False,
    heuristic_inflation_factor: float = 1,
) -> List[Cost]:
    costs = create_scaled_l1norm_position_continuity_costs(
        u_vars, v_vars, heuristic_inflation_factor
    )

    if add_const_cost:
        total_dims = u_vars.all.size + v_vars.all.size
        # Constant cost for the edge
        a = np.zeros((total_dims, 1))
        # We add 2 because if a shortcut is used it minimally replaces 2 edges
        constant_cost = 2 * heuristic_inflation_factor
        costs.append(LinearCost(a, constant_cost))

    return costs


def contact_shortcut_edge_l2norm_cost_factory_obj_weighted(
    u_vars: ContactSetDecisionVariables,
    v_vars: ContactSetDecisionVariables,
    add_const_cost: bool = False,
    heuristic_inflation_factor: float = 1,
) -> List[Cost]:
    costs = create_scaled_l2norm_position_continuity_costs(
        u_vars, v_vars, heuristic_inflation_factor
    )

    if add_const_cost:
        total_dims = u_vars.all.size + v_vars.all.size
        # Constant cost for the edge
        a = np.zeros((total_dims, 1))
        # We add 2 because if a shortcut is used it minimally replaces 2 edges
        constant_cost = 2 * heuristic_inflation_factor
        costs.append(LinearCost(a, constant_cost))

    return costs


def contact_shortcut_edge_l1_norm_plus_switches_cost_factory(
    u_vars: ContactSetDecisionVariables,
    v_vars: ContactSetDecisionVariables,
    n_switches: int,
    heuristic_inflation_factor: float = 1,
) -> List[Cost]:
    """Assumes that simultaneous mode switches are not allowed."""
    costs = create_scaled_l1norm_position_continuity_costs(
        u_vars, v_vars, heuristic_inflation_factor
    )

    total_dims = u_vars.all.size + v_vars.all.size
    # Constant cost for the edge
    a = np.zeros((total_dims, 1))
    constant_cost = (1 + n_switches) * heuristic_inflation_factor
    costs.append(LinearCost(a, constant_cost))

    return costs


### VERTEX COST CREATION ###


def contact_vertex_cost_position_l2norm(
    vars: ContactSetDecisionVariables, scaling: float = 1.0
) -> L2NormCost:
    """Creates a vertex cost that penalizes the length of the path in position
    space.

    vars.pos has shape (Euclidean/base dim, num positions/pos order per
    set) So to get the path length we need to diff over the second axis.
    """
    exprs = np.diff(vars.pos).flatten() * scaling
    A = DecomposeLinearExpressions(exprs, vars.all)
    b = np.zeros(A.shape[0])
    # print(f"vertex_cost_position_path_length A: {A}")
    return L2NormCost(A, b)


def contact_vertex_cost_position_l1norm(
    vars: ContactSetDecisionVariables, scaling: float = 1.0
) -> L1NormCost:
    """Creates a vertex cost that penalizes the l1 norm of the path in position
    space.

    vars.pos has shape (Euclidean/base dim, num positions/pos order per
    set) So to get the path length we need to diff over the second axis.
    """
    exprs = np.diff(vars.pos).flatten() * scaling
    A = DecomposeLinearExpressions(exprs, vars.all)
    b = np.zeros(A.shape[0])
    # print(f"vertex_cost_position_path_length A: {A}")
    return L1NormCost(A, b)


### VERTEX CONSTRAINT CREATION ###


def contact_vertex_constraint_last_pos_equality_contact(
    vars: ContactSetDecisionVariables, sample: np.ndarray
) -> LinearEqualityConstraint:
    """Creates a constraint that enforces the last position of the vertex to be
    the same as those elements in the sample.

    Size of vars should be the same as the size of the sample.
    """
    exprs = vars.last_pos - vars.last_pos_from_all(sample)
    # Decompose affine expression into expr = Ax + b
    A, b = DecomposeAffineExpressions(exprs, vars.all)
    # Linear equality constraint of the form: Ax = -b
    return LinearEqualityConstraint(A, -b)


def contact_vertex_constraint_last_pos_eps_equality(
    vars: ContactSetDecisionVariables, sample: np.ndarray, eps: float = 1e-3
) -> LinearConstraint:
    """Creates a constraint that enforces the last position of the vertex to be
    within eps of those elements in the sample.

    Size of vars should be the same as the size of the sample.
    """
    exprs = vars.last_pos - vars.last_pos_from_all(sample)
    # Decompose affine expression into expr = Ax + b
    A, b = DecomposeAffineExpressions(exprs, vars.all)
    # Linear equality constraint of the form: Ax = -b
    return LinearConstraint(A, -b - eps, -b + eps)


def contact_vertex_constraint_eps_bounding_box(
    sample: np.ndarray, eps: float = 1e-3
) -> BoundingBoxConstraint:
    """Creates a constraint that enforces the last position of the vertex to be
    within eps of those elements in the sample.

    Size of vars should be the same as the size of the sample.
    """
    lb = sample - eps
    ub = sample + eps
    return BoundingBoxConstraint(lb, ub)


### EDGE COST CREATION ###


def contact_edge_cost_constant(
    u_vars: ContactSetDecisionVariables,
    v_vars: ContactSetDecisionVariables,
    constant_cost: float = 1,
) -> LinearCost:
    """Creates a cost that penalizes each active edge a constant value."""
    total_dims = u_vars.all.size + v_vars.all.size
    # Linear cost of the form: a'x + b, where a is a vector of coefficients and b is a constant.
    a = np.zeros((total_dims, 1))
    b = constant_cost
    return LinearCost(a, b)


### EDGE CONSTRAINT CREATION ###


def contact_edge_constraint_position_continuity(
    u_vars: ContactSetDecisionVariables, v_vars: ContactSetDecisionVariables
) -> LinearEqualityConstraint:
    """Creates a constraint that enforces position continuity between the last
    position in vertex u to the first position in vertex v, given there's an
    edge from u to v.

    Since this is an edge constraint, the decision variables will be
    those of both the u and v vertices.
    """
    u_vars_all = create_vars_from_template(u_vars.all, "u")
    v_vars_all = create_vars_from_template(v_vars.all, "v")
    uv_vars_all = np.concatenate((u_vars_all, v_vars_all))
    exprs = (
        u_vars.last_pos_from_all(u_vars_all) - v_vars.first_pos_from_all(v_vars_all)
    ).flatten()

    # Decompose affine expression into expr = Ax + b
    A, b = DecomposeAffineExpressions(exprs, uv_vars_all)
    # Linear equality constraint of the form: Ax = -b
    return LinearEqualityConstraint(A, -b)
