import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import Hyperellipsoid

from large_gcs.geometry.convex_set import ConvexSet


class Ellipsoid(ConvexSet):
    """
    Wrapper for the Drake HyperEllipsoid class representing: {x | (x-center)ᵀAᵀA(x-center) ≤ 1}.
    """

    def __init__(self, center, A):
        """
        Default constructor for the ellipsoid {x | (x-center)ᵀAᵀA(x-center) ≤ 1}.
        Args:
            center: (m,1) or (m,) numpy array or list.
            A: (m,n) numpy array or list.
        """
        self.A = np.array(A).astype("float64")
        self._center = np.array(center).astype("float64")
        reshaped_center = self._center.reshape(self.A.shape[0], 1)
        self._hyper_ellipsoid = Hyperellipsoid(self.A, reshaped_center)

    def _plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        B = self.A.T @ self.A
        l, v = np.linalg.eig(B)
        angle = 180 * np.arctan2(*v[0]) / np.pi + 90
        ellipse = (self.center, 2 * l[0] ** -0.5, 2 * l[1] ** -0.5, angle)
        patch = patches.Ellipse(*ellipse, **kwargs)
        ax.add_artist(patch)

    @property
    def dim(self):
        return self._center.size

    @property
    def set(self):
        return self._hyper_ellipsoid

    @property
    def center(self):
        return self._center
