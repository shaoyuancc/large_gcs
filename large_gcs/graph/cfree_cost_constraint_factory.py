from typing import Optional

import numpy as np
from pydrake.all import L2NormCost, LinearEqualityConstraint

EPS_SCALING_FACTOR = 10


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
    A = A * EPS_SCALING_FACTOR
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
