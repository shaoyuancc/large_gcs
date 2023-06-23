import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import (HPolyhedron, VPolytope)
from large_gcs.geometry.convex_set import ConvexSet


class Polyhedron(ConvexSet):
    """
    Wrapper for the Drake HPolyhedron class that uses the half-space representation: {x| A x ≤ b}
    """

    def __init__(self, vertices):
        """
        Default constructor for the polyhedron {x| A x ≤ b}.
        Args:
            list of vertices.
        """
        self._vertices = np.array(vertices).astype('float64')
        # Verify that the vertices are in the same dimension
        assert len(set([v.size for v in self._vertices])) == 1

        self._dimension = self._vertices.shape[1]
        self._h_polyhedron = HPolyhedron(VPolytope(vertices))

    def _plot(self, **kwargs):
        if self.vertices.shape[0] < 3:
            raise NotImplementedError
        plt.fill(*self.vertices.T, **kwargs)
    
    @property
    def vertices(self):
        return self._vertices
    