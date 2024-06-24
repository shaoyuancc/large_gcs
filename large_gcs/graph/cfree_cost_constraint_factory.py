import numpy as np
import scipy
from pydrake.all import L2NormCost, LinearEqualityConstraint


def create_cfree_shortcut_edge_factory_under(
    u_dim: int, add_const_cost: bool = False
) -> L2NormCost:
    """Assumes u is a Polyhedron with 2 knot points, and v is a Point with only
    1 knot point."""
    if add_const_cost:
        raise NotImplementedError()

    base_dim = u_dim // 2
    A = np.hstack((np.zeros((base_dim, base_dim)), np.eye(base_dim), -np.eye(base_dim)))
    b = np.zeros((base_dim,))
    return [L2NormCost(A, b)]


def create_cfree_shortcut_edge_factory_over(
    u_dim: int, add_const_cost: bool = False
) -> L2NormCost:
    if add_const_cost:
        raise NotImplementedError()
    base_dim = u_dim // 2
    A = np.hstack((np.zeros((base_dim, base_dim)), np.eye(base_dim), -np.eye(base_dim)))
    b = np.zeros((base_dim,))
    EPS_SCALING_FACTOR = 100
    A = A * EPS_SCALING_FACTOR
    return [L2NormCost(A, b)]


def create_cfree_shortcut_edge_factory_translation_weighted_over(
    u_dim: int, add_const_cost: bool = False
) -> L2NormCost:
    # Assumes 3 translation dim and 3 rotation dim
    if add_const_cost:
        raise NotImplementedError()
    base_dim = 6
    trans_dim = 3
    trans_weighted_sub_A = scipy.linalg.block_diag(
        np.eye(trans_dim) * 100, np.eye(trans_dim) * 10
    )
    A = np.hstack(
        (np.zeros((base_dim, base_dim)), trans_weighted_sub_A, -trans_weighted_sub_A)
    )
    b = np.zeros((base_dim,))
    return [L2NormCost(A, b)]


def create_cfree_shortcut_edge_factory_xy_translation_weighted_over(
    u_dim: int, add_const_cost: bool = False
) -> L2NormCost:
    # Assumes 3 translation dim and 3 rotation dim
    if add_const_cost:
        raise NotImplementedError()
    base_dim = 6
    trans_weighted_sub_A = scipy.linalg.block_diag(np.eye(2) * 100, np.eye(4) * 10)
    A = np.hstack(
        (np.zeros((base_dim, base_dim)), trans_weighted_sub_A, -trans_weighted_sub_A)
    )
    b = np.zeros((base_dim,))
    return [L2NormCost(A, b)]


def create_cfree_l2norm_vertex_cost(base_dim: int) -> L2NormCost:
    A = np.hstack((np.eye(base_dim), -np.eye(base_dim)))
    b = np.zeros((base_dim,))
    return L2NormCost(A, b)


def create_cfree_continuity_edge_constraint(
    offset: np.ndarray,
) -> LinearEqualityConstraint:
    """ux1 + offset == vx0, where the edge variables are: [ux0, ux1, vx0,
    vx1]"""
    base_dim = offset.shape[0]
    A = np.hstack(
        (
            np.zeros((base_dim, base_dim)),
            np.eye(base_dim),
            -np.eye(base_dim),
            np.zeros((base_dim, base_dim)),
        )
    )
    b = -offset
    return LinearEqualityConstraint(A, b)


def create_source_region_edge_constraint(base_dim: int) -> LinearEqualityConstraint:
    A = np.hstack((np.eye(base_dim), -np.eye(base_dim), np.zeros((base_dim, base_dim))))
    b = np.zeros((base_dim,))
    return LinearEqualityConstraint(A, b)


def create_region_target_edge_constraint(base_dim: int) -> LinearEqualityConstraint:
    A = np.hstack((np.zeros((base_dim, base_dim)), np.eye(base_dim), -np.eye(base_dim)))
    b = np.zeros((base_dim,))
    return LinearEqualityConstraint(A, b)


def vertex_constraint_last_pos_equality_cfree(
    sample: np.ndarray,
) -> LinearEqualityConstraint:
    """Assumes that the convex set has 2 knot points and the last position is
    the second knot point i.e. the second half of the variables."""
    base_dim = sample.shape[0] // 2
    A = np.hstack((np.zeros((base_dim, base_dim)), np.eye(base_dim)))
    b = sample[base_dim:]
    return LinearEqualityConstraint(A, b)
