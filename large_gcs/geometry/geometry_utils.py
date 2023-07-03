import numpy as np


def is_on_hyperplane(a, b, x):
    """Returns whether x is on the hyperplane defined by ax = b.
    a and x are vectors with the same dimension, b is a scalar.
    """
    return np.isclose(np.dot(a, x), b)


def counter_clockwise_angle_between(v1, v2):
    """Returns the counter-clockwise angle between two vectors."""
    assert len(v1) == len(v2) == 2, "Vectors must be 2D"
    return np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
