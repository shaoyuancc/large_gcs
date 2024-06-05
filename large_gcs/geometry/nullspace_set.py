import logging

import numpy as np
from pydrake.all import AffineSubspace
from pydrake.all import ConvexSet as DrakeConvexSet
from pydrake.all import HPolyhedron
from pydrake.all import Point as DrakePoint

from large_gcs.geometry.convex_set import ConvexSet
from large_gcs.geometry.geometry_utils import remove_rows_near_zero

logger = logging.getLogger(__name__)
AFFINE_SUBSPACE_TOL = 1e-9


class NullspaceSet(ConvexSet):
    """A Polyhedron has a NullspaceSet which is either a Drake HPolyhedron or
    Point."""

    def __init__(self, drake_set):
        self._set = drake_set

    @classmethod
    def from_hpolyhedron(
        cls: "NullspaceSet",
        h_polyhedron: HPolyhedron,
        should_reduce_inequalities: bool = False,
    ):
        # Find affine subspace of H x <= h
        affine_subspace = AffineSubspace(h_polyhedron, tol=AFFINE_SUBSPACE_TOL)
        V = affine_subspace.basis()

        # Check whether the affine subspace is a point
        if V.shape[1] == 0:
            return cls.from_point(DrakePoint(affine_subspace.translation()))

        x_0 = affine_subspace.translation()
        H, h = h_polyhedron.A(), h_polyhedron.b()
        A_prime = H @ V
        b_prime = h - H @ x_0

        A_prime, b_prime = remove_rows_near_zero(
            A_prime, b_prime, tol=AFFINE_SUBSPACE_TOL
        )

        hpoly = HPolyhedron(A_prime, b_prime)
        if should_reduce_inequalities:
            # logger.debug(f"A_prime before: {A_prime.shape}")
            hpoly = hpoly.ReduceInequalities()
            # logger.debug(f"A_prime after: {self._set.A().shape}")
        ns_set = cls(hpoly)
        ns_set._V = V
        ns_set._x_0 = x_0
        return ns_set

    @classmethod
    def from_point(cls, point: DrakePoint):
        ns_set = cls(point)
        ns_set._V = np.zeros((point.ambient_dimension(), point.ambient_dimension()))
        ns_set._x_0 = point.x()
        return ns_set

    def get_samples(self, n_samples=100) -> np.ndarray:
        if isinstance(self._set, DrakePoint):
            return np.array([self._set.x()])
        q_samples = super().get_samples(n_samples)
        p_samples = q_samples @ self._V.T + self._x_0
        return p_samples

    @property
    def dim(self):
        return self._set.ambient_dimension()

    @property
    def set(self) -> DrakeConvexSet:
        return self._set

    @property
    def center(self):
        if isinstance(self._set, DrakePoint):
            return self._set.x()
        return self._set.ChebyshevCenter()

    @property
    def V(self):
        return self._V

    @property
    def x_0(self):
        return self._x_0
