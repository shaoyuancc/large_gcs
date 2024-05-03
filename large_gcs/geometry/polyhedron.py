import itertools
import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy
from pydrake.all import (
    DecomposeAffineExpressions,
    Formula,
    FormulaKind,
    HPolyhedron,
    Variable,
    VPolytope,
)
from scipy.spatial import ConvexHull

from large_gcs.geometry.convex_set import ConvexSet
from large_gcs.geometry.geometry_utils import is_on_hyperplane
from large_gcs.utils.utils import copy_pastable_str_from_np_array

logger = logging.getLogger(__name__)


class Polyhedron(ConvexSet):
    """
    Wrapper for the Drake HPolyhedron class that uses the half-space representation: {x| A x ≤ b}
    """

    def __init__(self, A, b, should_compute_vertices=True):
        """
        Default constructor for the polyhedron {x| A x ≤ b}.
        """
        self._vertices = None
        self._center = None

        A = np.array(A)
        b = np.array(b)

        # Detect and remove rows with very small A and b, practically zero \leq zero
        for i, (a1, b1) in enumerate(zip(A, b)):
            if np.allclose(a1, 0) and np.isclose(b1, 0):
                # logger.debug(
                #     f"Removing row {i} from A and b, because they are practically zero."
                # )
                A = np.delete(A, i, axis=0)
                b = np.delete(b, i)

        if Polyhedron._check_contains_equality_constraints(A, b):
            self._has_equality_constraints = True
            self._h_polyhedron = HPolyhedron(A, b)
            if not self._h_polyhedron.IsEmpty():
                self._create_null_space_polyhedron()
            return
        else:
            self._has_equality_constraints = False

        if A.shape[1] == 1:
            self._h_polyhedron = HPolyhedron(A, b)
            return

        if should_compute_vertices:
            vertices = VPolytope(HPolyhedron(A, b)).vertices().T
            hull = ConvexHull(vertices)  # orders vertices counterclockwise
            self._vertices = vertices[hull.vertices]
            A, b = Polyhedron._reorder_A_b_by_vertices(A, b, self._vertices)

        self._h_polyhedron = HPolyhedron(A, b)

        # Compute center
        if should_compute_vertices:
            try:
                max_ellipsoid = self._h_polyhedron.MaximumVolumeInscribedEllipsoid()
                self._center = np.array(max_ellipsoid.center())
            except:
                logger.warning("Could not compute center")
                self._center = None

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
        polyhedron = cls(h_polyhedron.A(), h_polyhedron.b())
        if polyhedron._vertices is None:
            polyhedron._vertices = vertices
            # Set center to be the mean of the vertices
            polyhedron._center = np.mean(vertices, axis=0)
        return polyhedron

    @classmethod
    def from_constraints(cls, constraints: List[Formula], variables: List[Variable]):
        """
        Construct a polyhedron from a list of constraint formulas.
        Args:
            constraints: array of constraint formulas.
            variables: array of variables.
        """

        # In case the constraints or variables were multi-dimensional lists
        constraints = np.concatenate([c.flatten() for c in constraints])
        variables = np.concatenate([v.flatten() for v in variables])

        expressions = []
        for formula in constraints:
            kind = formula.get_kind()
            lhs, rhs = formula.Unapply()[1]
            if kind == FormulaKind.Eq:
                # Eq constraint ax = b is
                # implemented as ax ≤ b, -ax <= -b
                expressions.append(lhs - rhs)
                expressions.append(rhs - lhs)
            elif kind == FormulaKind.Geq:
                # lhs >= rhs
                # ==> rhs - lhs ≤ 0
                expressions.append(rhs - lhs)
            elif kind == FormulaKind.Leq:
                # lhs ≤ rhs
                # ==> lhs - rhs ≤ 0
                expressions.append(lhs - rhs)

        # We now have expr ≤ 0 for all expressions
        # ==> we get Ax - b ≤ 0
        A, b_neg = DecomposeAffineExpressions(expressions, variables)

        # Polyhedrons are of the form: Ax <= b
        b = -b_neg
        polyhedron = cls(A, b)

        polyhedron.constraints = constraints
        polyhedron.variables = variables
        polyhedron.expressions = expressions

        return polyhedron

    def _plot(self, **kwargs):
        if self.dim == 1:
            # Add extra dimension to vertices for plotting
            vertices = np.hstack((self.vertices, np.zeros((self.vertices.shape[0], 1))))
        else:
            vertices = self.vertices
        plt.fill(*vertices.T, **kwargs)

    def plot_vertex(self, index, **kwargs):
        assert index < self.vertices.shape[0], "Index out of bounds"
        if self.dim == 1:
            vertex = np.array([self.vertices[index], 0])
        else:
            vertex = self.vertices[index]
        plt.scatter(*vertex, **kwargs)
        plt.annotate(
            index,
            vertex,
            textcoords="offset points",
            xytext=(5, 5),
            ha="center",
        )

    def plot_face(self, index, **kwargs):
        assert index < self.vertices.shape[0], "Index out of bounds"
        vertices = np.array(
            [self.vertices[index], self.vertices[(index + 1) % self.vertices.shape[0]]]
        )
        plt.plot(*vertices.T, **kwargs)

    @staticmethod
    def _reorder_A_b_by_vertices(A, b, vertices):
        """
        Reorders the halfspace representation A x ≤ b so that they follow the same order as the vertices.
        i.e. the first row of A and the first element of b correspond to the line between the first and second vertices.
        """
        # assert len(A) == len(vertices) == len(b)
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

    @staticmethod
    def _check_contains_equality_constraints(A, b, rtol=1e-5, atol=1e-8):
        """Equality constraints are enforced by having one row in A and b be: ax ≤ b and another row be: -ax ≤ -b.
        So checking if any pairs of rows add up to 0 tells us whether there are any equality constraints.
        """
        for (i, (a1, b1)), (j, (a2, b2)) in itertools.combinations(
            enumerate(zip(A, b)), 2
        ):
            if np.allclose(a1 + a2, 0, rtol=rtol, atol=atol) and np.isclose(
                b1 + b2, 0, rtol=rtol, atol=atol
            ):
                return True
        return False

    def has_equality_constraints(self):
        """Equality constraints are enforced by having one row in A and b be: ax ≤ b and another row be: -ax ≤ -b.
        So checking if any pairs of rows add up to 0 tells us whether there are any equality constraints.
        """
        return self._check_contains_equality_constraints(self.set.A(), self.set.b())

    @staticmethod
    def get_separated_inequality_equality_constraints(
        A_original, b_original, rtol=1e-5, atol=1e-8
    ):
        """Separate and return A, b, C, d where A x ≤ b are inequalities and C x = d are equalities."""

        equality_indices = set()
        equality_rows = []

        for (i1, (a1, b1)), (i2, (a2, b2)) in itertools.combinations(
            enumerate(zip(A_original, b_original)), 2
        ):
            if np.allclose(a1 + a2, 0, rtol=rtol, atol=atol) and np.isclose(
                b1 + b2, 0, rtol=rtol, atol=atol
            ):
                # logger.debug(f"Equality constraints found: {i1}, {i2}")
                # logger.debug(f"a1: {a1}, b1: {b1}")
                # logger.debug(f"a2: {a2}, b2: {b2}")
                equality_indices.update([i1, i2])
                equality_rows.append(i1)
        # logger.debug(f"equality_indices: {equality_indices}")
        # logger.debug(f"equality_rows: {equality_rows}")
        C = np.array([A_original[i] for i in equality_rows])
        d = np.array([b_original[i] for i in equality_rows])

        inequality_rows = [
            (a, b)
            for i, (a, b) in enumerate(zip(A_original, b_original))
            if i not in equality_indices
        ]
        A_ineq, b_ineq = (
            zip(*inequality_rows)
            if inequality_rows
            else (np.empty((0, A_original.shape[1])), np.array([]))
        )
        return np.array(A_ineq), np.array(b_ineq), C, d

    def _create_null_space_polyhedron(self):
        # logger.debug(f"A_original = {print_copy_pastable_np_array(self.set.A())}")
        # logger.debug(f"b_original = {print_copy_pastable_np_array(self.set.b())}")

        # Separate original A and B into inequality and equality constraints
        A, b, C, d = self.get_separated_inequality_equality_constraints(
            self.set.A(), self.set.b()
        )

        # logger.debug(
        #     f"\n A.shape: {A.shape}, b.shape: {b.shape}, C.shape: {C.shape}, d.shape: {d.shape}"
        # )
        # logger.debug(
        #     f"ranks: A: {np.linalg.matrix_rank(A)}, C: {np.linalg.matrix_rank(C)}"
        # )
        # logger.debug(f"C = {copy_pastable_str_from_np_array(C)}")
        # logger.debug(f"d = {copy_pastable_str_from_np_array(d)}")

        # Compute the basis of the null space of C
        self._V = scipy.linalg.null_space(C)
        # # Compute the pseudo-inverse of C
        # C_pinv = np.linalg.pinv(C)

        # # Use the pseudo-inverse to find x_0
        # self._x_0 = np.dot(C_pinv, d)
        self._x_0, residuals, rank, s = np.linalg.lstsq(C, d, rcond=None)
        A_prime = A @ self._V
        b_prime = b - A @ self._x_0

        self._null_space_polyhedron = Polyhedron(
            A=A_prime, b=b_prime, should_compute_vertices=False
        )

    def get_samples(self, n_samples=100):
        if self._has_equality_constraints:
            q_samples = self._null_space_polyhedron.get_samples(n_samples)
            assert len(q_samples) > 0
            p_samples = q_samples @ self._V.T + self._x_0

            return p_samples
        else:
            return super().get_samples(n_samples)

    @property
    def dim(self):
        return self.set.A().shape[1]

    @property
    def set(self):
        return self._h_polyhedron

    # The following properties rely on vertices and center being set,
    # they will not work for polyhedra with equality constraints.

    @property
    def bounding_box(self):
        return np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)])

    @property
    def vertices(self):
        return self._vertices

    @property
    def center(self):
        return self._center
