import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import HPolyhedron, VPolytope
from scipy.spatial import ConvexHull
from large_gcs.geometry.convex_set import ConvexSet


class Polyhedron(ConvexSet):
    """
    Wrapper for the Drake HPolyhedron class that uses the half-space representation: {x| A x ≤ b}
    """

    def __init__(self, A, b):
        """
        Default constructor for the polyhedron {x| A x ≤ b}.
        """
        self._h_polyhedron = HPolyhedron(A, b)
        print(f"Polyhedron is bounded {self._h_polyhedron.IsBounded()}")
        self._vertices = VPolytope(self._h_polyhedron).vertices().T
        hull = ConvexHull(self._vertices)  # orders vertices counterclockwise
        self._vertices = self._vertices[hull.vertices]

        # Compute center
        max_ellipsoid = self._h_polyhedron.MaximumVolumeInscribedEllipsoid()
        self._center = np.array(max_ellipsoid.center())

    @classmethod
    def from_vertices(cls, vertices):
        """
        Construct a polyhedron from a list of vertices.
        Args:
            list of vertices.
        """
        vertices = np.array(vertices)
        # Verify that the vertices are in the same dimension
        assert len(set([v.size for v in vertices])) == 1

        v_polytope = VPolytope(vertices.T)
        h_polyhedron = HPolyhedron(v_polytope)

        return cls(h_polyhedron.A(), h_polyhedron.b())

    def _plot(self, **kwargs):
        if self.vertices.shape[0] < 3:
            raise NotImplementedError

        plt.fill(*self.vertices.T, **kwargs)

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
