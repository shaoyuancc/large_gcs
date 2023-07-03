import numpy as np
import matplotlib.pyplot as plt
from pydrake.all import HPolyhedron, VPolytope
from scipy.spatial import ConvexHull
from large_gcs.geometry.convex_set import ConvexSet
from large_gcs.geometry.geometry_utils import *


class Polyhedron(ConvexSet):
    """
    Wrapper for the Drake HPolyhedron class that uses the half-space representation: {x| A x ≤ b}
    """

    def __init__(self, A, b):
        """
        Default constructor for the polyhedron {x| A x ≤ b}.
        """
        vertices = VPolytope(HPolyhedron(A, b)).vertices().T
        hull = ConvexHull(vertices)  # orders vertices counterclockwise
        self._vertices = vertices[hull.vertices]
        A, b = Polyhedron._reorder_A_b_by_vertices(A, b, self._vertices)
        self._h_polyhedron = HPolyhedron(A, b)

        # Compute center
        max_ellipsoid = self._h_polyhedron.MaximumVolumeInscribedEllipsoid()
        self._center = np.array(max_ellipsoid.center())

    @staticmethod
    def _reorder_A_b_by_vertices(A, b, vertices):
        """
        Reorders the halfspace representation A x ≤ b so that they follow the same order as the vertices.
        i.e. the first row of A and the first element of b correspond to the line between the first and second vertices.
        """
        assert len(A) == len(vertices) == len(b)
        new_order = []
        for i in range(len(vertices)):
            for j in range(len(vertices)):
                if is_on_hyperplane(A[j], b[j], vertices[i]) and is_on_hyperplane(
                    A[j], b[j], vertices[(i + 1) % len(vertices)]
                ):
                    new_order.append(j)
                    break
        assert len(new_order) == len(vertices), "Something went wrong"
        return A[new_order], b[new_order]

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

    def plot_vertex(self, index, **kwargs):
        assert index < self.vertices.shape[0], "Index out of bounds"
        plt.scatter(*self.vertices[index], **kwargs)

    def plot_face(self, index, **kwargs):
        assert index < self.vertices.shape[0], "Index out of bounds"
        vertices = np.array(
            [self.vertices[index], self.vertices[(index + 1) % self.vertices.shape[0]]]
        )
        plt.plot(*vertices.T, **kwargs)

    @property
    def bounding_box(self):
        return np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)])

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
