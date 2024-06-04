import logging

import numpy as np
from pydrake.all import AffineSubspace, HPolyhedron
from pydrake.all import Point as DrakePoint

from large_gcs.geometry.convex_set import ConvexSet

logger = logging.getLogger(__name__)
AFFINE_SUBSPACE_TOL = 1e-3


class NullspaceSet(ConvexSet):
    """A Polyhedron has a NullspaceSet which is either a Drake HPolyhedron or
    Point."""

    def __init__(
        self, h_polyhedron: HPolyhedron, should_reduce_inequalities: bool = False
    ):
        # Find affine subspace of H x <= h

        self._affine_subspace = AffineSubspace(h_polyhedron, tol=AFFINE_SUBSPACE_TOL)
        V = self._affine_subspace.basis()

        # Check whether the affine subspace is a point
        if V.shape[1] == 0:
            self._set = DrakePoint(self._affine_subspace.translation())
            return
        self._V = V
        self._x_0 = self._affine_subspace.translation()
        H, h = h_polyhedron.A(), h_polyhedron.b()
        A_prime = H @ self._V
        b_prime = h - H @ self._x_0

        # Create a boolean mask to identify rows to delete
        delete_mask = np.zeros(len(A_prime), dtype=bool)
        # Detect rows with very small A
        for i, (a1, b1) in enumerate(zip(A_prime, b_prime)):
            if np.allclose(a1, 0, atol=AFFINE_SUBSPACE_TOL):
                delete_mask[i] = True
        # Filter out the rows to be deleted
        A_prime = A_prime[~delete_mask]
        b_prime = b_prime[~delete_mask]

        self._set = HPolyhedron(A_prime, b_prime)
        if should_reduce_inequalities:
            # logger.debug(f"A_prime before: {A_prime.shape}")
            self._set = self._set.ReduceInequalities()
            # logger.debug(f"A_prime after: {self._set.A().shape}")

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
    def set(self):
        return self._set

    @property
    def center(self):
        if isinstance(self._set, DrakePoint):
            return self._set.x()
        return self._set.ChebyshevCenter()
