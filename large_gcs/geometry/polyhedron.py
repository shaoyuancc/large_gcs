import itertools
from typing import List

import matplotlib.pyplot as plt
import numpy as np
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


class Polyhedron(ConvexSet):
    """
    Wrapper for the Drake HPolyhedron class that uses the half-space representation: {x| A x ≤ b}
    """

    def __init__(self, A, b):
        """
        Default constructor for the polyhedron {x| A x ≤ b}.
        """
        A = np.array(A)
        b = np.array(b)

        if Polyhedron._has_equality_constraints(A, b) or A.shape[1] == 1:
            self._h_polyhedron = HPolyhedron(A, b)
            self._vertices = None
            self._center = None
            return

        vertices = VPolytope(HPolyhedron(A, b)).vertices().T
        hull = ConvexHull(vertices)  # orders vertices counterclockwise
        self._vertices = vertices[hull.vertices]
        A, b = Polyhedron._reorder_A_b_by_vertices(A, b, self._vertices)
        self._h_polyhedron = HPolyhedron(A, b)

        # Compute center
        try:
            max_ellipsoid = self._h_polyhedron.MaximumVolumeInscribedEllipsoid()
            self._center = np.array(max_ellipsoid.center())
        except:
            print("Could not compute center")
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

        return cls(h_polyhedron.A(), h_polyhedron.b())

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
        # if self.vertices.shape[0] < 3:
        #     raise NotImplementedError

        plt.fill(*self.vertices.T, **kwargs)

    def plot_vertex(self, index, **kwargs):
        assert index < self.vertices.shape[0], "Index out of bounds"
        plt.scatter(*self.vertices[index], **kwargs)
        plt.annotate(
            index,
            self.vertices[index],
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
    def _has_equality_constraints(A, b):
        """Equality constraints are enforced by having one row in A and b be: ax ≤ b and another row be: -ax ≤ -b.
        So checking if any pairs of rows add up to 0 tells us whether there are any equality constraints.
        """
        for (a1, b1), (a2, b2) in itertools.product(zip(A, b), zip(A, b)):
            if np.isclose(a1 + a2, [0] * len(a1)).all() and np.isclose(b1 + b2, 0):
                return True
        return False

    def has_equality_constraints(self):
        """Equality constraints are enforced by having one row in A and b be: ax ≤ b and another row be: -ax ≤ -b.
        So checking if any pairs of rows add up to 0 tells us whether there are any equality constraints.
        """
        return self._has_equality_constraints(self.set.A(), self.set.b())

    def get_separated_inequality_equality_constraints(self):
        """Separate and return A, b, C, d where A x ≤ b are inequalities and C x = d are equalities."""
        A_original = self.set.A()
        b_original = self.set.b()

        equality_rows = []
        inequality_rows = []

        # Mark rows involved in equalities to avoid re-adding them as inequalities
        marked_for_equality = set()

        for (i1, (a1, b1)) in enumerate(zip(A_original, b_original)):
            for (i2, (a2, b2)) in enumerate(zip(A_original, b_original)):
                if (
                    i1 < i2
                    and np.isclose(a1 + a2, [0] * len(a1)).all()
                    and np.isclose(b1 + b2, 0)
                ):
                    equality_rows.append((i1, i2))
                    marked_for_equality.update([i1, i2])

        C = np.array([A_original[i] for i, _ in equality_rows])
        d = np.array([b_original[i] for i, _ in equality_rows])

        # Collect rows not marked for equality as inequalities
        for i, (a, b) in enumerate(zip(A_original, b_original)):
            if i not in marked_for_equality:
                inequality_rows.append((a, b))

        A_ineq, b_ineq = (
            zip(*inequality_rows)
            if inequality_rows
            else (np.empty((0, A_original.shape[1])), np.array([]))
        )

        return np.array(A_ineq), np.array(b_ineq), C, d

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
