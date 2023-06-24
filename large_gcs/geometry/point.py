import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pydrake.all import Point as DrakePoint
from large_gcs.geometry.convex_set import ConvexSet


class Point(ConvexSet):
    """
    Wrapper for Drake Point Class. Convex set containing exactly one element.
    """

    def __init__(self, x):
        self._point = DrakePoint(x)

    def _plot(self, **kwargs):
        plt.scatter(*self.center, c="k", **kwargs)

    @property
    def dimension(self):
        return self.center.size

    @property
    def set(self):
        return self._point

    @property
    def center(self):
        return self._point.x()
