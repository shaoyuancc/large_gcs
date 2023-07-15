from dataclasses import dataclass
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
from pydrake.all import (
    MakeMatrixContinuousVariable,
    MakeVectorContinuousVariable,
    Variable,
    eq,
    ge,
)
from large_gcs.geometry.polyhedron import Polyhedron
from copy import copy


class MobilityType(Enum):
    STATIC = 1
    UNACTUATED = 2
    ACTUATED = 3


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

        self._create_decision_vars()
        self._create_constraints()

    def _create_decision_vars(self):
        if self.mobility_type != MobilityType.STATIC:
            # Decision variables for positions
            self.vars_pos = MakeMatrixContinuousVariable(
                self.dim, self.n_pos_points, self.name + "_pos"
            )
            # Expressions for velocities in terms of positions
            self.vars_vel = self.vars_pos[:, 1:] - self.vars_pos[:, 0:-1]
            """
            Note that the above is a simplified model, where time for each segment is assumed to be 1.
            Before putting this onto the real robot, we will need to add a time variable for each segment.
            Also note that the number of columns in vars_pos is one more than the number of columns in vars_vel.
            """
            self.vars_pos_x = self.vars_pos[0, :]
            self.vars_pos_y = self.vars_pos[1, :]

            # Decision variables for forces
            # Resultant force on the body
            self.vars_force_res = MakeVectorContinuousVariable(
                self.dim, self.name + "_force_res"
            )

            # Actuation force on/of the body
            if self.mobility_type == MobilityType.ACTUATED:
                self.vars_force_act = MakeVectorContinuousVariable(
                    self.dim, self.name + "_force_act"
                )

    def _create_constraints(self):
        constraints = []
        if self.mobility_type != MobilityType.STATIC:
            # Force constraints
            for vel in self.vars_vel.T:
                # Ensures that the resultant force is in the same direction as the velocity,
                # and that the velocity is 0 if the resultant force is 0
                constraints.extend(eq(self.vars_force_res, vel).tolist())

            eps = 1e-3

        self.constraints = constraints

    @property
    def dim(self):
        """Dimension of the underlying geometry of the body, not the dimension of the configuration space"""
        return self.geometry.dim

    @property
    def n_vertices(self):
        return len(self.geometry.vertices)

    @property
    def n_faces(self):
        return len(self.geometry.set.b())

    def plot(self):
        plt.rc("axes", axisbelow=True)
        plt.gca().set_aspect("equal")
        self.geometry.plot()
        plt.text(*self.geometry.center, self.name, ha="center", va="center")
        self._plot_vertices()
        self._plot_face_labels()

    def _plot_vertices(self, **kwargs):
        for i in range(self.n_vertices):
            plt.text(
                *self._get_offset_pos(self.geometry.vertices[i]),
                f"v{i}",
                ha="center",
                va="center",
            )

    def _plot_face_labels(self):
        vertices = np.vstack([self.geometry.vertices, self.geometry.vertices[0]])
        for i in range(self.n_faces):
            mid = (vertices[i] + vertices[i + 1]) / 2
            plt.text(*self._get_offset_pos(mid), f"f{i}", ha="center", va="center")

    def _get_offset_pos(self, pos):
        offset_dir = pos - self.geometry.center
        offset_hat = offset_dir / np.linalg.norm(offset_dir)
        return pos + offset_hat * 0.1

    def plot_at_position(self, pos, **kwargs):
        vertices = self.geometry.vertices
        p_CT = pos - self.geometry.center
        vertices_shifted = vertices + p_CT
        plt.fill(*vertices_shifted.T, **kwargs)
        plt.text(*pos, self.name, ha="center", va="center")
