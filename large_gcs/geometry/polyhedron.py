import logging
from typing import List, Type

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from pydrake.all import (
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

        # If the ambient dimension of the polyhedron is 2
        if vertices.shape[1] == 2:
            vertices = order_vertices_counter_clockwise(vertices)

        v_polytope = VPolytope(vertices.T)
        h_polyhedron = HPolyhedron(v_polytope)
        if vertices.shape[1] == 2:
            H, h = Polyhedron._reorder_A_b_by_vertices(
                h_polyhedron.A(), h_polyhedron.b(), vertices
            )
        else:
            H, h = h_polyhedron.A(), h_polyhedron.b()

        polyhedron = cls(H, h, should_compute_vertices=False)
        if polyhedron._vertices is None:
            polyhedron._vertices = vertices
            # Set center to be the mean of the vertices
            polyhedron._center = np.mean(vertices, axis=0)

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

    def plot_transformation(self, T, t=None, **kwargs):
        if t is None:
            t = np.zeros((T.shape[0],))
        transformed_vertices = self.vertices @ T.T + t
        # orders vertices counterclockwise
        hull = ConvexHull(transformed_vertices)
        if transformed_vertices.shape[1] == 2:
            if "name" in kwargs:
                kwargs["label"] = kwargs["name"]
                del kwargs["name"]
            if "fig" in kwargs:
                del kwargs["fig"]
            kwargs["linestyle"] = "dotted"
            transformed_vertices = transformed_vertices[hull.vertices]
            # Repeat the first vertex to close the polygon
            transformed_vertices = np.vstack(
                (transformed_vertices, transformed_vertices[0])
            )
            # print(f"transformed_vertices: {transformed_vertices}")
            plt.plot(*transformed_vertices.T, **kwargs)
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("Transformed Polyhedron")
            return plt
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
                    opacity=0.3,
                    showlegend=True,
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
