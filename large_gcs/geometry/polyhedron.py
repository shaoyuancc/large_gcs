import itertools
import logging
from typing import List, Type

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scipy
from pydrake.all import (
    AffineSubspace,
    DecomposeAffineExpressions,
    Formula,
    FormulaKind,
    HPolyhedron,
    VPolytope,
)
from scipy.spatial import ConvexHull

from large_gcs.geometry.convex_set import ConvexSet
from large_gcs.geometry.geometry_utils import (
    is_on_hyperplane,
    order_vertices_counter_clockwise,
)
from large_gcs.geometry.nullspace_set import NullspaceSet

logger = logging.getLogger(__name__)


class Polyhedron(ConvexSet):
    """Wrapper for the Drake HPolyhedron class that uses the half-space
    representation: {x| H x ≤ h}"""

    def __init__(
        self, H: np.ndarray, h: np.ndarray, should_compute_vertices: bool = True
    ):
        """Default constructor for the polyhedron {x| H x ≤ h}.

        This constructor should be kept cheap to run since many
        polyhedrons are constructed and then thrown away if they are
        empty or they don't intersect other sets.
        """
        self._vertices = None
        self._center = None

        self._h_polyhedron = HPolyhedron(H, h)
        self._H = H
        self._h = h

        if should_compute_vertices:
            if H.shape[1] == 1 or self._h_polyhedron.IsEmpty():
                logger.warning("Polyhedron is empty or 1D, skipping compute vertices")
                return

            self._vertices = order_vertices_counter_clockwise(
                VPolytope(self._h_polyhedron).vertices().T
            )
            H, h = Polyhedron._reorder_A_b_by_vertices(H, h, self._vertices)

            self._h_polyhedron = HPolyhedron(H, h)
            self._H = H
            self._h = h

            # Compute center
            try:
                max_ellipsoid = self._h_polyhedron.MaximumVolumeInscribedEllipsoid()
                self._center = np.array(max_ellipsoid.center())
            except:
                logger.warning("Could not compute center")
                self._center = None

    def create_nullspace_set(self):
        # logger.debug(f"H size before: {self._h_polyhedron.A().shape}")
        # self._h_polyhedron = self._h_polyhedron.ReduceInequalities(tol=0)
        # logger.debug(f"H size after: {self._h_polyhedron.A().shape}")
        if self._h_polyhedron.IsEmpty():
            logger.warning("Polyhedron is empty, skipping nullspace set creation")
            return
        self._nullspace_set = NullspaceSet.from_hpolyhedron(
            self._h_polyhedron, should_reduce_inequalities=False
        )

    @classmethod
    def from_vertices(cls, vertices):
        """Construct a polyhedron from a list of vertices.

        Args:
            list of vertices.
        """
        vertices = np.array(vertices)
        # Verify that the vertices are in the same dimension
        assert len(set([v.size for v in vertices])) == 1

        vertices = order_vertices_counter_clockwise(vertices)

        v_polytope = VPolytope(vertices.T)
        h_polyhedron = HPolyhedron(v_polytope)
        H, h = Polyhedron._reorder_A_b_by_vertices(
            h_polyhedron.A(), h_polyhedron.b(), vertices
        )

        polyhedron = cls(H, h, should_compute_vertices=False)
        if polyhedron._vertices is None:
            polyhedron._vertices = vertices
            # Set center to be the mean of the vertices
            polyhedron._center = np.mean(vertices, axis=0)

        (
            polyhedron._A,
            polyhedron._b,
            polyhedron._C,
            polyhedron._d,
        ) = Polyhedron.get_separated_inequality_equality_constraints(
            h_polyhedron.A(), h_polyhedron.b()
        )
        polyhedron.create_nullspace_set()
        return polyhedron

    @classmethod
    def from_constraints(
        cls: Type["Polyhedron"], constraints: List[Formula], variables: np.ndarray
    ):
        """Construct a polyhedron from a list of constraint formulas.

        Args:
            constraints: array of constraint formulas.
            variables: array of variables.
        """
        A, b, C, d = None, None, None, None
        ineq_expr = []
        eq_expr = []
        for formula in constraints:
            kind = formula.get_kind()
            lhs, rhs = formula.Unapply()[1]
            if kind == FormulaKind.Eq:
                # Eq constraint lhs = rhs ==> lhs - rhs = 0
                eq_expr.append(lhs - rhs)
            elif kind == FormulaKind.Geq:
                # lhs >= rhs
                # ==> rhs - lhs ≤ 0
                ineq_expr.append(rhs - lhs)
            elif kind == FormulaKind.Leq:
                # lhs ≤ rhs
                # ==> lhs - rhs ≤ 0
                ineq_expr.append(lhs - rhs)

        # We now have expr ≤ 0 for all inequality expressions
        # ==> we get Ax - b ≤ 0
        if ineq_expr:
            A, b_neg = DecomposeAffineExpressions(ineq_expr, variables)
            b = -b_neg
            # logger.debug(f"Decomposed inequality constraints: A = {A}, b = {b}")
        if eq_expr:
            C, d_neg = DecomposeAffineExpressions(eq_expr, variables)
            d = -d_neg
            # logger.debug(f"Decomposed equality constraints: C = {C}, d = {d}")

        if ineq_expr and eq_expr:
            # Rescaled Matrix H, and vector h
            H = np.vstack((A, C, -C))
            h = np.concatenate((b, d, -d))
            polyhedron = cls(H, h, should_compute_vertices=False)
        elif ineq_expr:
            polyhedron = cls(A, b, should_compute_vertices=False)
        elif eq_expr:
            polyhedron = cls(C, d, should_compute_vertices=False)
        else:
            raise ValueError("No constraints given")
        # Store the separated inequality and equality constraints
        polyhedron._A = A
        polyhedron._b = b
        polyhedron._C = C
        polyhedron._d = d

        return polyhedron

    def _plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        if self.dim == 1:
            # Add extra dimension to vertices for plotting
            vertices = np.hstack((self.vertices, np.zeros((self.vertices.shape[0], 1))))
        else:
            vertices = self.vertices
        ax.fill(*vertices.T, **kwargs)

    def transform_vertices(self, T, t):
        logger.debug(
            f"T.shape: {T.shape}, t.shape: {t.shape}, vertices.shape: {self.vertices.shape}"
        )
        transformed_vertices = self.vertices @ T.T + t
        return transformed_vertices

    def plot_transformation(self, T, t, **kwargs):
        transformed_vertices = self.vertices @ T.T + t
        # orders vertices counterclockwise
        hull = ConvexHull(transformed_vertices)
        if transformed_vertices.shape[1] == 2:
            transformed_vertices = transformed_vertices[hull.vertices]
            # Repeat the first vertex to close the polygon
            transformed_vertices = np.vstack(
                (transformed_vertices, transformed_vertices[0])
            )
            # print(f"transformed_vertices: {transformed_vertices}")
            plt.plot(*transformed_vertices.T, **kwargs)
            plt.title("Transformed Polyhedron")
        elif transformed_vertices.shape[1] == 3:
            # MATPLOTLIB

            # # Setting up the plot
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            # # Collect the vertices for each face of the convex hull
            # faces = [transformed_vertices[simplex] for simplex in hull.simplices]
            # face_collection = Poly3DCollection(faces, **kwargs)
            # ax.add_collection3d(face_collection)

            # # Set the limits for the axes
            # for coord in range(3):
            #     lim = (np.min(transformed_vertices[:, coord]), np.max(transformed_vertices[:, coord]))
            #     getattr(ax, f'set_xlim' if coord == 0 else f'set_ylim' if coord == 1 else f'set_zlim')(lim)

            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')

            # Plotly
            # if 'fig' not in kwargs:
            if "fig" not in kwargs:
                # Creating the plot
                fig = go.Figure()
            else:
                fig = kwargs["fig"]
                del kwargs["fig"]

            # Adding each face of the convex hull to the plot
            # print(f"number of simplices: {len(hull.simplices)}")
            # print(f"simplices: {hull.simplices}")
            # for simplex in hull.simplices:
            #     fig.add_trace(go.Mesh3d(
            #         x=transformed_vertices[simplex, 0],
            #         y=transformed_vertices[simplex, 1],
            #         z=transformed_vertices[simplex, 2],
            #         flatshading=True,
            #         **kwargs
            #     ))

            # Extracting the vertices for each face of the convex hull
            x, y, z = transformed_vertices.T

            # Creating the plot
            fig = fig.add_trace(
                go.Mesh3d(
                    x=x,
                    y=y,
                    z=z,
                    i=hull.simplices[:, 0],
                    j=hull.simplices[:, 1],
                    k=hull.simplices[:, 2],
                    **kwargs,
                )
            )

            # Wireframe
            # for simplex in hull.simplices:
            #     # Plot each edge of the simplex (triangle)
            #     for i in range(len(simplex)):
            #         # Determine start and end points for each line segment
            #         start, end = simplex[i], simplex[(i+1) % len(simplex)]
            #         fig.add_trace(go.Scatter3d(
            #             x=[x[start], x[end]],
            #             y=[y[start], y[end]],
            #             z=[z[start], z[end]],
            #             mode='lines',
            #             line=kwargs
            #         ))

            # Setting plot layout
            fig.update_layout(
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                title="Transformed Polyhedron",
            )

            return fig
        else:
            raise ValueError("Cannot plot polyhedron with more than 3 dimensions")

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
        """Reorders the halfspace representation A x ≤ b so that they follow
        the same order as the vertices.

        i.e. the first row of A and the first element of b correspond to
        the line between the first and second vertices.
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
        """Equality constraints are enforced by having one row in A and b be:
        ax ≤ b and another row be: -ax ≤ -b.

        So checking if any pairs of rows add up to 0 tells us whether
        there are any equality constraints.
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
        """Equality constraints are enforced by having one row in A and b be:
        ax ≤ b and another row be: -ax ≤ -b.

        So checking if any pairs of rows add up to 0 tells us whether
        there are any equality constraints.
        """
        return self._check_contains_equality_constraints(self.set.A(), self.set.b())

    @staticmethod
    def get_separated_inequality_equality_constraints(
        A_original, b_original, rtol=1e-5, atol=1e-8
    ):
        """Separate and return A, b, C, d where A x ≤ b are inequalities and C
        x = d are equalities."""
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

    def get_samples(self, n_samples=100):
        return self._nullspace_set.get_samples(n_samples)

    @property
    def dim(self):
        return self.set.A().shape[1]

    @property
    def set(self):
        return self._h_polyhedron

    @property
    def H(self):
        return self._H

    @property
    def h(self):
        return self._h

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @property
    def C(self):
        return self._C

    @property
    def d(self):
        return self._d

    @property
    def nullspace_set(self):
        return self._nullspace_set

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
