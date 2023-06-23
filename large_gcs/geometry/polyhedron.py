import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import (HPolyhedron, VPolytope)
from scipy.spatial import ConvexHull
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

        v_polytope = VPolytope(self._vertices.T)
        self._h_polyhedron = HPolyhedron(v_polytope)

        # Compute center
        max_ellipsoid = self._h_polyhedron.MaximumVolumeInscribedEllipsoid()
        self._center = np.array(max_ellipsoid.center())

    def _plot(self, **kwargs):
        if self.vertices.shape[0] < 3:
            raise NotImplementedError
        hull = ConvexHull(self.vertices) # orders vertices counterclockwise
        vertices = self.vertices[hull.vertices]
        plt.fill(*vertices.T, **kwargs)
    
    @property
    def vertices(self):
        return self._vertices
    
    @property
    def dimension(self):
        return self._vertices.shape[1]
    
    @property
    def set(self):
        return self._h_polyhedron
    
    @property
    def center(self):
        return self._center
    