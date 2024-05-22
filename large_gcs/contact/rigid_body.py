from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import (
    MakeMatrixContinuousVariable,
    MakeVectorContinuousVariable,
    eq,
    ge,
    le,
)

from large_gcs.geometry.polyhedron import Polyhedron


class MobilityType(Enum):
    STATIC = 1
    UNACTUATED = 2
    ACTUATED = 3


BodyColor = {
    "object": "lightsalmon",
    "robot": "lightblue",
    "target": "lightgreen",
}


@dataclass
class RigidBodyParams:
    name: str
    vertices: np.ndarray
    mobility_type: MobilityType
    n_pos_points: int


@dataclass
class RigidBody:
    name: str
    geometry: Polyhedron  # For now, only allow convex sets, and specifically polyhedra
    mobility_type: MobilityType
    # Position is piecewise linear segments, order is the number of points (pos_order - 1 = number of segments)
    n_pos_points: int = 2

    def __post_init__(self):
        if self.geometry.dim != 2:
            raise NotImplementedError

        if self.n_pos_points != 2:
            raise NotImplementedError

        self._create_decision_vars()
        # self._create_force_constraints()

    def from_params(params: RigidBodyParams):
        return RigidBody(
            params.name,
            Polyhedron.from_vertices(params.vertices),
            params.mobility_type,
            params.n_pos_points,
        )

    def _create_decision_vars(self):
        if self.mobility_type != MobilityType.STATIC:
            # Decision variables for positions
            self.vars_pos = MakeMatrixContinuousVariable(
                self.dim, self.n_pos_points, self.name + "_pos"
            )
            # Expressions for velocities in terms of positions
            self.vars_vel = (self.vars_pos[:, 1:] - self.vars_pos[:, 0:-1]).flatten()
            """Note that the above is a simplified model, where time for each
            segment is assumed to be 1.

            Before putting this onto the real robot, we will need to add
            a time variable for each segment. Also note that the number
            of columns in vars_pos is one more than the number of
            columns in vars_vel.
            """
            self.vars_pos_x = self.vars_pos[0, :]
            self.vars_pos_y = self.vars_pos[1, :]

            # Actuation force on/of the body
            if self.mobility_type == MobilityType.ACTUATED:
                self.vars_force_act = MakeVectorContinuousVariable(
                    self.dim, self.name + "_force_act"
                )

    def create_workspace_position_constraints(self, workspace):
        self.base_workspace_constraints = self._create_workspace_position_constraints(
            workspace, base_only=True
        )
        self.workspace_constraints = self._create_workspace_position_constraints(
            workspace, base_only=False
        )

    def _create_workspace_position_constraints(self, workspace, base_only=False):
        constraints = []
        ws = workspace.T
        bbox = self.geometry.bounding_box
        # Position constraints
        if base_only:
            positions = [self.vars_pos.T[0]]
        else:
            positions = self.vars_pos.T
        for pos in positions:
            ub_offset = bbox[1] - self.geometry.center
            constraints.extend(le(pos + ub_offset, ws[1]).tolist())
            lb_offset = bbox[0] - self.geometry.center
            constraints.extend(ge(pos + lb_offset, ws[0]).tolist())
        return constraints

    @property
    def dim(self):
        """Dimension of the underlying geometry of the body, not the dimension
        of the configuration space."""
        return self.geometry.dim

    @property
    def n_vertices(self):
        return len(self.geometry.vertices)

    @property
    def n_faces(self):
        return len(self.geometry.set.b())

    @property
    def params(self):
        return RigidBodyParams(
            self.name, self.geometry.vertices, self.mobility_type, self.n_pos_points
        )

    @property
    def vars_base_pos(self):
        return self.vars_pos[:, 0]

    def plot(self):
        plt.rc("axes", axisbelow=True)
        plt.gca().set_aspect("equal")
        self.geometry.plot()
        plt.text(*self.geometry.center, self.name, ha="center", va="center")
        self._plot_vertices()
        self._plot_face_labels()

    def _plot_vertices(self, pos=None, ax=None, **kwargs):
        # Use the provided axis or get the current axis
        if ax is None:
            ax = plt.gca()

        verts = self.geometry.vertices
        center = self.geometry.center
        if pos is not None:
            p_CT = pos - self.geometry.center
            center = pos
            verts = verts + p_CT
        for i in range(self.n_vertices):
            ax.text(
                *self._get_offset_pos(verts[i], center),
                f"v{i}",
                ha="center",
                va="center",
            )

    def _plot_face_labels(self, pos=None, ax=None, **kwargs):
        # Use the provided axis or get the current axis
        if ax is None:
            ax = plt.gca()

        verts = self.geometry.vertices
        center = self.geometry.center
        if pos is not None:
            center = pos
            p_CT = pos - self.geometry.center
            verts = verts + p_CT
        verts = np.vstack([verts, verts[0]])
        for i in range(self.n_faces):
            mid = (verts[i] + verts[i + 1]) / 2
            ax.text(
                *self._get_offset_pos(mid, center), f"f{i}", ha="center", va="center"
            )

    def _get_offset_pos(self, pos, center):
        offset_dir = pos - center
        offset_hat = offset_dir / np.linalg.norm(offset_dir)
        return pos + offset_hat * 0.15

    def plot_at_com(
        self, label_body=True, label_vertices_faces=False, ax=None, **kwargs
    ) -> None:
        self.plot_at_position(
            self.geometry.center, label_body, label_vertices_faces, ax=ax, **kwargs
        )

    def plot_at_position(
        self, pos, label_body=True, label_vertices_faces=False, ax=None, **kwargs
    ) -> None:
        # Use the provided axis or get the current axis
        if ax is None:
            ax = plt.gca()

        vertices_shifted = self.get_vertices_at_position(pos)
        ax.fill(*vertices_shifted.T, **kwargs)
        if label_body:
            ax.text(*pos, self.name, ha="center", va="center")  # type: ignore
        if label_vertices_faces:
            self._plot_vertices(pos, ax)
            self._plot_face_labels(pos, ax)

    def get_vertices_at_position(self, pos):
        p_CT = pos - self.geometry.center
        return self.geometry.vertices + p_CT
