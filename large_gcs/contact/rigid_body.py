from dataclasses import dataclass
from enum import Enum
import numpy as np
from pydrake.all import MakeMatrixContinuousVariable
from large_gcs.geometry.polyhedron import Polyhedron


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

    @property
    def dim(self):
        return self.geometry.dim

    @property
    def n_vertices(self):
        return len(self.geometry.vertices)

    @property
    def n_faces(self):
        return len(self.geometry.set.b())
