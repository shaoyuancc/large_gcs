import numpy as np


def is_on_hyperplane(a, b, x):
    """Returns whether x is on the hyperplane defined by ax = b.
    a and x are vectors with the same dimension, b is a scalar.
    """
    return np.isclose(np.dot(a, x), b)
