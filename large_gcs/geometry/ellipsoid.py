import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pydrake.all import (HyperEllipsoid, VPolytope)
from large_gcs.geometry.convex_set import ConvexSet

class Ellipsoid(ConvexSet):
    """
    Wrapper for the Drake HyperEllipsoid class representing: {x | (x-center)ᵀAᵀA(x-center) ≤ 1}.
    """

    def __init__(self, A, center):
        """
        Default constructor for the ellipsoid {x | (x-center)ᵀAᵀA(x-center) ≤ 1}.
        Args:
            A: 2D numpy array.
            center: 1D numpy array.
        """
        self._A = np.array(A).astype('float64')
        self._center = np.array(center).astype('float64')
        self._dimension = self._center.size
        self._hyper_ellipsoid = HyperEllipsoid(self._A, self._center)

    def _plot(self, **kwargs):
        l, v = np.linalg.eig(self.A)
        angle = 180 * np.arctan2(*v[0]) / np.pi + 90
        ellipse = (self.center, 2 * l[0] ** -.5, 2 * l[1] ** -.5, angle)
        patch = patches.Ellipse(*ellipse, **kwargs)
        plt.gca().add_artist(patch)