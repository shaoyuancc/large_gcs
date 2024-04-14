from typing import List

import numpy as np
from pydrake.all import Cost, L2NormCost, LinearEqualityConstraint, QuadraticCost


def shortcut_edge_cost_factory(dim: int) -> List[Cost]:
    edge_cost = create_l2norm_edge_cost(dim)
    return [edge_cost]


def create_l2norm_edge_cost(dim: int):
    A = np.hstack((np.eye(dim), -np.eye(dim)))
    b = np.zeros((dim, 1))
    edge_cost = L2NormCost(A, b)
    return edge_cost


"""
There seems to be a bug that causes gcs solve convex restrictions to fail sometimes when using this cost
e.g. hor vert graph, Without create_2d_x_equality_edge_constraint on edge ('s', 'p0')
["('s', 'p0')", "('p0', 't')"] fails
"""
def create_l2norm_squared_edge_cost(dim: int):
    I_n = np.identity(dim)
    Q = np.block([[I_n, -I_n], [-I_n, I_n]])
    b = np.zeros((2 * dim, 1))
    c = 0
    edge_cost = QuadraticCost(Q, b, c)
    return edge_cost


def create_l2norm_vertex_cost_from_point(point: np.ndarray):
    A = np.eye(point.shape[0])
    b = -point
    vertex_cost = L2NormCost(A, b)
    return vertex_cost


def create_l2norm_vertex_cost(dim: int):
    A = np.eye(dim)
    b = np.zeros((dim, 1))
    edge_cost = L2NormCost(A, b)
    return edge_cost


# There seems to be a bug where this causes gcs solve convex restrictions to fail sometimes
def create_l2norm_squared_vertex_cost_from_point(point: np.ndarray):
    """
    ||x - point||^2 = (x - point)^T (x - point) = x^T x - 2 x^T point + point^T point
    QuadraticCost(Q, b, c) = 0.5 x^T Q x + b^T x + c
    So, Q = 2 * I, b = -2 * point, c = point^T point
    """
    Q = 2 * np.identity(point.shape[0])
    b = -2 * point
    c = point.T @ point
    vertex_cost = QuadraticCost(Q, b, c)
    return vertex_cost


def create_l2norm_squared_vertex_cost(dim: int):
    Q = np.identity(dim)
    b = np.zeros((dim, 1))
    c = 0
    edge_cost = QuadraticCost(Q, b, c)
    return edge_cost


def create_2d_x_equality_edge_constraint():
    A = np.array([[1, 0, -1, 0]])
    b = np.zeros(1)
    return LinearEqualityConstraint(A, b)


def create_2d_y_equality_edge_constraint():
    A = np.array([[0, 1, 0, -1]])
    b = np.zeros(1)
    return LinearEqualityConstraint(A, b)


def create_equality_edge_constraint(dim: int):
    A = np.hstack((np.eye(dim), -np.eye(dim)))
    b = np.zeros((dim, 1))
    return LinearEqualityConstraint(A, b)
