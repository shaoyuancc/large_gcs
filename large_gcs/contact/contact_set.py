from typing import List

import numpy as np
from pydrake.all import Formula
from pydrake.all import Point as DrakePoint

from large_gcs.contact.contact_pair_mode import ContactPairMode
from large_gcs.contact.contact_set_decision_variables import ContactSetDecisionVariables
from large_gcs.contact.rigid_body import MobilityType, RigidBody
from large_gcs.geometry.convex_set import ConvexSet
from large_gcs.geometry.geometry_utils import HPolyhedronFromConstraints
from large_gcs.geometry.nullspace_set import NullspaceSet
from large_gcs.geometry.polyhedron import Polyhedron


class ContactPointSet(ConvexSet):
    def __init__(
        self,
        id: str,
        objects: List[RigidBody],
        robots: List[RigidBody],
        object_positions: List[np.ndarray],
        robot_positions: List[np.ndarray],
    ):
        assert len(objects) == len(object_positions)
        assert len(robots) == len(robot_positions)
        positions = np.array(object_positions + robot_positions)

        self.vars = ContactSetDecisionVariables.base_vars_from_objs_robs(
            objects, robots
        )
        self._point = DrakePoint(positions.flatten())
        self._nullspace_set = NullspaceSet.from_point(self._point)

        self._id = id

    @property
    def id(self):
        return self._id

    @property
    def dim(self):
        return self.set.ambient_dimension()

    @property
    def set(self):
        return self._point

    @property
    def base_set(self):
        return self._point

    @property
    def center(self):
        return self._point.x()

    @property
    def A(self):
        return None

    @property
    def b(self):
        return None

    @property
    def C(self):
        return np.eye(self.dim)

    @property
    def d(self):
        return self._point.x()

    @property
    def nullspace_set(self):
        return self._nullspace_set


class ContactSet(ConvexSet):
    def __init__(
        self,
        vars: ContactSetDecisionVariables,
        contact_pair_modes: List[ContactPairMode],
        additional_constraints: List[Formula] = None,
        additional_base_constraints: List[Formula] = None,
    ):
        self.vars = vars

        self.contact_pair_modes = contact_pair_modes
        self.constraint_formulas = [
            constraint
            # constraint.item()
            for mode in contact_pair_modes
            for constraint in mode.constraint_formulas
        ]
        self.base_constraint_formulas = [
            constraint
            # constraint.item()
            for mode in contact_pair_modes
            for constraint in mode.base_constraint_formulas
        ]
        if additional_constraints is not None:
            self.constraint_formulas.extend(additional_constraints)
        if additional_base_constraints is not None:
            self.base_constraint_formulas.extend(additional_base_constraints)

        self._polyhedron = Polyhedron.from_constraints(
            self.constraint_formulas, self.vars.all
        )
        self._base_polyhedron = HPolyhedronFromConstraints(
            self.base_constraint_formulas,
            self.vars.base_all,
        )

    @classmethod
    def from_objs_robs(
        cls,
        contact_pair_modes: List[ContactPairMode],
        objects: List[RigidBody],
        robots: List[RigidBody],
        additional_constraints: List[Formula] = None,
        additional_base_constraints: List[Formula] = None,
    ):
        if not all(obj.mobility_type == MobilityType.UNACTUATED for obj in objects):
            raise ValueError("All objects must be unactuated")
        if not all(robot.mobility_type == MobilityType.ACTUATED for robot in robots):
            raise ValueError("All robots must be actuated")

        vars = ContactSetDecisionVariables.from_contact_pair_modes(
            objects, robots, contact_pair_modes
        )
        return cls(
            vars,
            contact_pair_modes,
            additional_constraints,
            additional_base_constraints,
        )

    def get_samples(self, n_samples=100):
        return self._polyhedron.get_samples(n_samples)

    @property
    def id(self):
        return tuple([mode.id for mode in self.contact_pair_modes])

    @property
    def dim(self):
        return self.set.A().shape[1]

    @property
    def set(self):
        # Note that self._polyhedron is a Polyhedron not Drake HPolyhedron
        return self._polyhedron.set

    @property
    def base_set(self):
        # Note that self._base_polyhedron is a Drake HPolyhedron
        return self._base_polyhedron

    @property
    def center(self):
        return None

    @property
    def H(self):
        return self._polyhedron.H

    @property
    def h(self):
        return self._polyhedron.h

    @property
    def A(self):
        return self._polyhedron.A

    @property
    def b(self):
        return self._polyhedron.b

    @property
    def C(self):
        return self._polyhedron.C

    @property
    def d(self):
        return self._polyhedron.d

    @property
    def nullspace_set(self):
        return self._polyhedron._nullspace_set
