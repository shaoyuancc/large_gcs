import numpy as np
from pydrake.all import L2NormCost, LinearEqualityConstraint, QuadraticCost


def create_l2norm_edge_cost(dim: int):
    A = np.hstack((np.eye(dim), -np.eye(dim)))
    b = np.zeros((dim, 1))
    edge_cost = L2NormCost(A, b)
    return edge_cost


def create_l2norm_squared_edge_cost(dim: int):
    I_n = np.identity(dim)
    Q = np.block([[I_n, -I_n], [-I_n, I_n]])
    b = np.zeros((2 * dim, 1))
    c = 0
    edge_cost = QuadraticCost(Q, b, c)
    return edge_cost


def create_l2norm_vertex_cost(dim: int):
    A = np.eye(dim)
    b = np.zeros((dim, 1))
    edge_cost = L2NormCost(A, b)
    return edge_cost


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
