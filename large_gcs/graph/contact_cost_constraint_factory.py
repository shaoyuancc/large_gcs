from typing import List

import numpy as np
from pydrake.all import (
    Cost,
    DecomposeAffineExpressions,
    DecomposeLinearExpressions,
    DecomposeQuadraticPolynomial,
    L1NormCost,
    L2NormCost,
    LinearConstraint,
    LinearCost,
    LinearEqualityConstraint,
    Polynomial,
    QuadraticCost,
    Variable,
)

from large_gcs.contact.contact_set_decision_variables import ContactSetDecisionVariables


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


def contact_shortcut_edge_cost_factory_under(
    u_vars: ContactSetDecisionVariables,
    v_vars: ContactSetDecisionVariables,
    add_const_cost: bool = False,
) -> List[Cost]:
    """Creates a list of costs for the shortcut between set u and set v"""
    u_vars_all = create_vars_from_template(u_vars.all, "u")
    v_vars_all = create_vars_from_template(v_vars.all, "v")

    # Position continuity cost
    u_pos = u_vars.pos_from_all(u_vars_all)
    v_pos = v_vars.pos_from_all(v_vars_all)
    u_last_pos = u_pos[:, :, -1].flatten()
    v_first_pos = v_pos[:, :, 0].flatten()
    uv_vars_all = np.concatenate((u_vars_all, v_vars_all))
    exprs = (u_last_pos - v_first_pos).flatten()
    A = DecomposeLinearExpressions(exprs, uv_vars_all)
    b = np.zeros(A.shape[0])
    costs = [L2NormCost(A, b)]

    if add_const_cost:
        # Constant cost for the edge
        a = np.zeros((uv_vars_all.size, 1))
        # We add 2 because if a shortcut is used it minimally replaces 2 edges
        constant_cost = 2
        costs.append(LinearCost(a, constant_cost))

    return costs


def contact_shortcut_edge_cost_factory_under_obj_weighted(
    u_vars: ContactSetDecisionVariables,
    v_vars: ContactSetDecisionVariables,
    add_const_cost: bool = False,
) -> List[Cost]:
    """Creates a list of costs for the shortcut between set u and set v"""
    u_vars_all = create_vars_from_template(u_vars.all, "u")
    v_vars_all = create_vars_from_template(v_vars.all, "v")
    uv_vars_all = np.concatenate((u_vars_all, v_vars_all))
    u_pos = u_vars.pos_from_all(u_vars_all)
    v_pos = v_vars.pos_from_all(v_vars_all)

    # Hacky way to separate the object and robot positions variables
    # I know that v will be the target, and so will not have force variables
    # I know that only robots have actuation force variables
    n_robs = u_vars.force_act.shape[0]
    n_objs = u_vars.pos.shape[0] - n_robs

    def create_l2norm_cost(u_pos, v_pos, scaling=1):
        u_last_pos = u_pos[:, :, -1].flatten()
        v_first_pos = v_pos[:, :, 0].flatten()
        exprs = (u_last_pos - v_first_pos).flatten() * scaling
        A = DecomposeLinearExpressions(exprs, uv_vars_all)
        b = np.zeros(A.shape[0])
        return L2NormCost(A, b)

    costs = [
        create_l2norm_cost(u_pos[:n_objs], v_pos[:n_objs], scaling=1),
        create_l2norm_cost(u_pos[n_objs:], v_pos[n_objs:], scaling=0.1),
    ]

    if add_const_cost:
        # Constant cost for the edge
        a = np.zeros((uv_vars_all.size, 1))
        # We add 2 because if a shortcut is used it minimally replaces 2 edges
        constant_cost = 2
        costs.append(LinearCost(a, constant_cost))

    return costs


def contact_shortcut_edge_l1_norm_cost_factory_under_obj_weighted(
    u_vars: ContactSetDecisionVariables,
    v_vars: ContactSetDecisionVariables,
    add_const_cost: bool = False,
) -> List[Cost]:
    """Creates a list of costs for the shortcut between set u and set v"""
    u_vars_all = create_vars_from_template(u_vars.all, "u")
    v_vars_all = create_vars_from_template(v_vars.all, "v")
    uv_vars_all = np.concatenate((u_vars_all, v_vars_all))
    u_pos = u_vars.pos_from_all(u_vars_all)
    v_pos = v_vars.pos_from_all(v_vars_all)

    # Hacky way to separate the object and robot positions variables
    # I know that v will be the target, and so will not have force variables
    # I know that only robots have actuation force variables
    n_robs = u_vars.force_act.shape[0]
    n_objs = u_vars.pos.shape[0] - n_robs

    def create_l1norm_cost(u_pos, v_pos, scaling=1):
        u_last_pos = u_pos[:, :, -1].flatten()
        v_first_pos = v_pos[:, :, 0].flatten()
        exprs = (u_last_pos - v_first_pos).flatten() * scaling
        A = DecomposeLinearExpressions(exprs, uv_vars_all)
        b = np.zeros(A.shape[0])
        return L1NormCost(A, b)

    costs = [
        create_l1norm_cost(u_pos[:n_objs], v_pos[:n_objs], scaling=1),
        create_l1norm_cost(u_pos[n_objs:], v_pos[n_objs:], scaling=0.1),
    ]

    if add_const_cost:
        # Constant cost for the edge
        a = np.zeros((uv_vars_all.size, 1))
        # We add 2 because if a shortcut is used it minimally replaces 2 edges
        constant_cost = 2
        costs.append(LinearCost(a, constant_cost))

    return costs


def contact_shortcut_edge_cost_factory_over(
    u_vars: ContactSetDecisionVariables,
    v_vars: ContactSetDecisionVariables,
    add_const_cost: bool = False,
) -> List[Cost]:
    """Creates a list of costs for the shortcut between set u and set v"""
    u_vars_all = create_vars_from_template(u_vars.all, "u")
    v_vars_all = create_vars_from_template(v_vars.all, "v")

    # Position continuity cost
    u_pos = u_vars.pos_from_all(u_vars_all)
    v_pos = v_vars.pos_from_all(v_vars_all)
    u_last_pos = u_pos[:, :, -1].flatten()
    v_first_pos = v_pos[:, :, 0].flatten()
    uv_vars_all = np.concatenate((u_vars_all, v_vars_all))
    exprs = (u_last_pos - v_first_pos).flatten()
    A = DecomposeLinearExpressions(exprs, uv_vars_all)
    b = np.zeros(A.shape[0])
    costs = [L2NormCost(A, b)]

    if add_const_cost:
        # Constant cost for the edge
        a = np.zeros((uv_vars_all.size, 1))
        # We add 2 because if a shortcut is used it minimally replaces 2 edges
        constant_cost = 2
        costs.append(LinearCost(a, constant_cost))

    return costs


def contact_norm_squared_shortcut_edge_cost_factory_over(
    u_vars: ContactSetDecisionVariables,
    v_vars: ContactSetDecisionVariables,
    add_const_cost: bool = False,
) -> List[Cost]:
    """Creates a list of costs for the shortcut between set u and set v"""
    u_vars_all = create_vars_from_template(u_vars.all, "u")
    v_vars_all = create_vars_from_template(v_vars.all, "v")

    # Position continuity cost
    u_pos = u_vars.pos_from_all(u_vars_all)
    v_pos = v_vars.pos_from_all(v_vars_all)
    u_last_pos = u_pos[:, :, -1].flatten()
    v_first_pos = v_pos[:, :, 0].flatten()
    uv_vars_all = np.concatenate((u_vars_all, v_vars_all))
    diff = (u_last_pos - v_first_pos).flatten()

    expr = np.dot(diff, diff)
    var_map = {var.get_id(): i for i, var in enumerate(uv_vars_all)}
    Q, b, c = DecomposeQuadraticPolynomial(Polynomial(expr), var_map)
    costs = [QuadraticCost(Q, b, c)]

    if add_const_cost:
        # Constant cost for the edge
        a = np.zeros((uv_vars_all.size, 1))
        # We add 2 because if a shortcut is used it minimally replaces 2 edges
        constant_cost = 2
        costs.append(LinearCost(a, constant_cost))

    return costs


def contact_shortcut_edge_cost_factory_over_obj_weighted(
    u_vars: ContactSetDecisionVariables,
    v_vars: ContactSetDecisionVariables,
    add_const_cost: bool = False,
) -> List[Cost]:
    """Creates a list of costs for the shortcut between set u and set v"""
    u_vars_all = create_vars_from_template(u_vars.all, "u")
    v_vars_all = create_vars_from_template(v_vars.all, "v")
    # Hacky way to separate the object and robot positions variables
    # I know that v will be the target, and so will not have force variables
    # I know that only robots have actuation force variables
    n_robs = u_vars.force_act.shape[0]
    n_objs = u_vars.pos.shape[0] - n_robs
    # Position continuity cost
    u_pos = u_vars.pos_from_all(u_vars_all)
    v_pos = v_vars.pos_from_all(v_vars_all)
    uv_vars_all = np.concatenate((u_vars_all, v_vars_all))

    def create_l2norm_cost(u_pos, v_pos, scaling=1):
        u_last_pos = u_pos[:, :, -1].flatten()
        v_first_pos = v_pos[:, :, 0].flatten()
        exprs = (u_last_pos - v_first_pos).flatten() * scaling
        A = DecomposeLinearExpressions(exprs, uv_vars_all)
        b = np.zeros(A.shape[0])
        return L2NormCost(A, b)

    costs = [
        create_l2norm_cost(u_pos[:n_objs], v_pos[:n_objs], scaling=10),
        create_l2norm_cost(u_pos[n_objs:], v_pos[n_objs:], scaling=2),
    ]

    if add_const_cost:
        # Constant cost for the edge
        a = np.zeros((uv_vars_all.size, 1))
        # We add 2 because if a shortcut is used it minimally replaces 2 edges
        constant_cost = 2
        costs.append(LinearCost(a, constant_cost))

    return costs


def contact_shortcut_edge_l1_norm_cost_factory_over_obj_weighted(
    u_vars: ContactSetDecisionVariables,
    v_vars: ContactSetDecisionVariables,
    add_const_cost: bool = False,
) -> List[Cost]:
    """Creates a list of costs for the shortcut between set u and set v"""
    u_vars_all = create_vars_from_template(u_vars.all, "u")
    v_vars_all = create_vars_from_template(v_vars.all, "v")
    # Hacky way to separate the object and robot positions variables
    # I know that v will be the target, and so will not have force variables
    # I know that only robots have actuation force variables
    n_robs = u_vars.force_act.shape[0]
    n_objs = u_vars.pos.shape[0] - n_robs
    # Position continuity cost
    u_pos = u_vars.pos_from_all(u_vars_all)
    v_pos = v_vars.pos_from_all(v_vars_all)
    uv_vars_all = np.concatenate((u_vars_all, v_vars_all))

    def create_l1norm_cost(u_pos, v_pos, scaling=1):
        u_last_pos = u_pos[:, :, -1].flatten()
        v_first_pos = v_pos[:, :, 0].flatten()
        exprs = (u_last_pos - v_first_pos).flatten() * scaling
        A = DecomposeLinearExpressions(exprs, uv_vars_all)
        b = np.zeros(A.shape[0])
        return L1NormCost(A, b)

    costs = [
        create_l1norm_cost(u_pos[:n_objs], v_pos[:n_objs], scaling=10),
        create_l1norm_cost(u_pos[n_objs:], v_pos[n_objs:], scaling=2),
    ]

    if add_const_cost:
        # Constant cost for the edge
        a = np.zeros((uv_vars_all.size, 1))
        # We add 2 because if a shortcut is used it minimally replaces 2 edges
        constant_cost = 2
        costs.append(LinearCost(a, constant_cost))

    return costs


def contact_norm_squared_shortcut_edge_cost_factory_over_obj_weighted(
    u_vars: ContactSetDecisionVariables,
    v_vars: ContactSetDecisionVariables,
    add_const_cost: bool = False,
) -> List[Cost]:
    """Creates a list of costs for the shortcut between set u and set v"""
    u_vars_all = create_vars_from_template(u_vars.all, "u")
    v_vars_all = create_vars_from_template(v_vars.all, "v")
    # Hacky way to separate the object and robot positions variables
    # I know that v will be the target, and so will not have force variables
    # I know that only robots have actuation force variables
    n_robs = u_vars.force_act.shape[0]
    n_objs = u_vars.pos.shape[0] - n_robs
    # Position continuity cost
    u_pos = u_vars.pos_from_all(u_vars_all)
    v_pos = v_vars.pos_from_all(v_vars_all)
    uv_vars_all = np.concatenate((u_vars_all, v_vars_all))

    def create_quadratic_cost(u_pos, v_pos, scaling=1):
        u_last_pos = u_pos[:, :, -1].flatten()
        v_first_pos = v_pos[:, :, 0].flatten()
        diff = (u_last_pos - v_first_pos).flatten()
        expr = np.dot(diff, diff) * scaling
        var_map = {var.get_id(): i for i, var in enumerate(uv_vars_all)}
        Q, b, c = DecomposeQuadraticPolynomial(Polynomial(expr), var_map)
        return QuadraticCost(Q, b, c)

    costs = [
        create_quadratic_cost(u_pos[:n_objs], v_pos[:n_objs], scaling=10),
        create_quadratic_cost(u_pos[n_objs:], v_pos[n_objs:], scaling=1),
    ]

    if add_const_cost:
        # Constant cost for the edge
        a = np.zeros((uv_vars_all.size, 1))
        # We add 2 because if a shortcut is used it minimally replaces 2 edges
        constant_cost = 2
        costs.append(LinearCost(a, constant_cost))

    return costs


### VERTEX COST CREATION ###


def vertex_cost_position_path_length(
    vars: ContactSetDecisionVariables, scaling: float = 1.0
) -> L2NormCost:
    """Creates a vertex cost that penalizes the length of the path in position space.
    vars.pos has shape (Euclidean/base dim, num positions/pos order per set)
    So to get the path length we need to diff over the second axis.
    """
    exprs = np.diff(vars.pos).flatten() * scaling
    A = DecomposeLinearExpressions(exprs, vars.all)
    b = np.zeros(A.shape[0])
    # print(f"vertex_cost_position_path_length A: {A}")
    return L2NormCost(A, b)


def vertex_cost_position_l1_norm(
    vars: ContactSetDecisionVariables, scaling: float = 1.0
) -> L1NormCost:
    """Creates a vertex cost that penalizes the l1 norm of the path in position space.
    vars.pos has shape (Euclidean/base dim, num positions/pos order per set)
    So to get the path length we need to diff over the second axis.
    """
    exprs = np.diff(vars.pos).flatten() * scaling
    A = DecomposeLinearExpressions(exprs, vars.all)
    b = np.zeros(A.shape[0])
    # print(f"vertex_cost_position_path_length A: {A}")
    return L1NormCost(A, b)


def vertex_cost_position_path_length_squared(
    vars: ContactSetDecisionVariables,
) -> QuadraticCost:
    path_length = np.diff(vars.pos).flatten()
    expr = np.dot(path_length, path_length)
    var_map = {var.get_id(): i for i, var in enumerate(vars.all)}
    Q, b, c = DecomposeQuadraticPolynomial(Polynomial(expr), var_map)
    return QuadraticCost(Q, b, c)


def vertex_cost_force_actuation_norm(vars: ContactSetDecisionVariables) -> L2NormCost:
    """Creates a vertex cost that penalizes the magnitude of the force actuation."""
    exprs = vars.force_act.flatten()
    A = DecomposeLinearExpressions(exprs, vars.all)
    b = np.zeros(A.shape[0])
    return L2NormCost(A, b)


def vertex_cost_force_actuation_norm_squared(
    vars: ContactSetDecisionVariables,
) -> QuadraticCost:
    """Creates a vertex cost that penalizes the magnitude of the force actuation squared."""
    expr = np.dot(vars.force_act.flatten(), vars.force_act.flatten())
    var_map = {var.get_id(): i for i, var in enumerate(vars.all)}
    Q, b, c = DecomposeQuadraticPolynomial(Polynomial(expr), var_map)
    return QuadraticCost(Q, b, c)


### VERTEX CONSTRAINT CREATION ###


def vertex_constraint_force_act_limits(
    vars: ContactSetDecisionVariables, lb: np.ndarray, ub: np.ndarray
) -> LinearConstraint:
    """Creates a constraint that limits the magnitude of the force actuation in each dimension."""
    assert vars.force_act.size > 0
    raise NotImplementedError


### EDGE COST CREATION ###


def edge_cost_constant(
    u_vars: ContactSetDecisionVariables,
    v_vars: ContactSetDecisionVariables,
    constant_cost: float = 1,
) -> LinearCost:
    """Creates a cost that penalizes each active edge a constant value."""
    u_vars_all = create_vars_from_template(u_vars.all, "u")
    v_vars_all = create_vars_from_template(v_vars.all, "v")
    # Linear cost of the form: a'x + b, where a is a vector of coefficients and b is a constant.
    uv_vars_all = np.concatenate((u_vars_all, v_vars_all))
    a = np.zeros((uv_vars_all.size, 1))
    b = constant_cost
    return LinearCost(a, b)


def edge_costs_position_continuity_norm(
    u_vars: ContactSetDecisionVariables,
    v_vars: ContactSetDecisionVariables,
    linear_scaling: float = 1,
) -> L2NormCost:
    # Get the last position in u and first position in v
    u_vars_all = create_vars_from_template(u_vars.all, "u")
    v_vars_all = create_vars_from_template(v_vars.all, "v")
    u_pos = u_vars.pos_from_all(u_vars_all)
    v_pos = v_vars.pos_from_all(v_vars_all)
    u_last_pos = u_pos[:, :, -1].flatten()
    v_first_pos = v_pos[:, :, 0].flatten()
    uv_vars_all = np.concatenate((u_vars_all, v_vars_all))
    exprs = (u_last_pos - v_first_pos).flatten() * linear_scaling
    A = DecomposeLinearExpressions(exprs, uv_vars_all)
    b = np.zeros(A.shape[0])
    return L2NormCost(A, b)


### EDGE CONSTRAINT CREATION ###


def edge_constraint_position_continuity(
    u_vars: ContactSetDecisionVariables, v_vars: ContactSetDecisionVariables
) -> LinearEqualityConstraint:
    """Creates a constraint that enforces position continuity between the last position in vertex u to
    the first position in vertex v, given there's an edge from u to v. Since this is an edge constraint,
    the decision variables will be those of both the u and v vertices.
    """
    # Get the last position in u and first position in v
    u_vars_all = create_vars_from_template(u_vars.all, "u")
    v_vars_all = create_vars_from_template(v_vars.all, "v")
    u_pos = u_vars.pos_from_all(u_vars_all)
    v_pos = v_vars.pos_from_all(v_vars_all)
    u_last_pos = u_pos[:, :, -1].flatten()
    v_first_pos = v_pos[:, :, 0].flatten()
    uv_vars_all = np.concatenate((u_vars_all, v_vars_all))
    exprs = (u_last_pos - v_first_pos).flatten()
    # Linear equality constraint of the form: Ax = b
    A, b = DecomposeAffineExpressions(exprs, uv_vars_all)
    return LinearEqualityConstraint(A, b)


def edge_constraint_position_continuity_factored(
    body_index, u_vars: ContactSetDecisionVariables, v_vars: ContactSetDecisionVariables
) -> LinearEqualityConstraint:
    """Creates a constraint that enforces position continuity between the last position in vertex u
    of body with body_index to the first position in vertex v (assuming v is the lower dimensional factored set,
    and u is the full dimensional set)
    """
    # Get the last position in u and first position in v
    u_vars_all = create_vars_from_template(u_vars.all, "u")
    v_vars_all = create_vars_from_template(v_vars.all, "v")
    u_pos = u_vars.pos_from_all(u_vars_all)
    v_pos = v_vars.pos_from_all(v_vars_all)
    u_last_pos = u_pos[body_index, :, -1].flatten()
    v_first_pos = v_pos[:, :, 0].flatten()
    assert u_last_pos.size == v_first_pos.size
    uv_vars_all = np.concatenate((u_vars_all, v_vars_all))
    exprs = (u_last_pos - v_first_pos).flatten()
    # Linear equality constraint of the form: Ax = b
    A, b = DecomposeAffineExpressions(exprs, uv_vars_all)
    return LinearEqualityConstraint(A, b)
