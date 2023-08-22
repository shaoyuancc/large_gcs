from typing import List

import numpy as np
from pydrake.all import Formula
from pydrake.all import Point as DrakePoint

from large_gcs.contact.contact_pair_mode import ContactPairMode, InContactPairMode
from large_gcs.contact.contact_set_decision_variables import ContactSetDecisionVariables
from large_gcs.contact.rigid_body import MobilityType, RigidBody
from large_gcs.geometry.convex_set import ConvexSet
from large_gcs.geometry.geometry_utils import HPolyhedronFromConstraints


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

        self.vars = ContactSetDecisionVariables.from_objs_robs(objects, robots)
        self._point = DrakePoint(positions.flatten())

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
        return None


class ContactSet(ConvexSet):
    def __init__(
        self,
        vars: ContactSetDecisionVariables,
        contact_pair_modes: List[ContactPairMode],
        additional_constraints: List[Formula] = None,
        remove_constraints_not_in_vars=False,
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
        self._polyhedron = HPolyhedronFromConstraints(
            self.constraint_formulas,
            self.vars.all,
            remove_constraints_not_in_vars=remove_constraints_not_in_vars,
        )
        self._base_polyhedron = HPolyhedronFromConstraints(
            self.base_constraint_formulas,
            self.vars.base_all,
            remove_constraints_not_in_vars=remove_constraints_not_in_vars,
        )

    @classmethod
    def from_objs_robs(
        cls,
        contact_pair_modes: List[ContactPairMode],
        objects: List[RigidBody],
        robots: List[RigidBody],
        additional_constraints: List[Formula] = None,
    ):

        if not all(obj.mobility_type == MobilityType.UNACTUATED for obj in objects):
            raise ValueError("All objects must be unactuated")
        if not all(robot.mobility_type == MobilityType.ACTUATED for robot in robots):
            raise ValueError("All robots must be actuated")

        vars = ContactSetDecisionVariables.from_contact_pair_modes(
            objects, robots, contact_pair_modes
        )
        return cls(vars, contact_pair_modes, additional_constraints, False)

    @classmethod
    def from_factored_collision_free_body(
        cls,
        contact_pair_modes: List[ContactPairMode],
        body: RigidBody,
        additional_constraints: List[Formula] = None,
    ):
        vars = ContactSetDecisionVariables.from_factored_collision_free_body(body)
        return cls(vars, contact_pair_modes, additional_constraints, True)

    @property
    def id(self):
        return tuple([mode.id for mode in self.contact_pair_modes])

    @property
    def dim(self):
        return self.set.A().shape[1]

    @property
    def set(self):
        return self._polyhedron

    @property
    def base_set(self):
        return self._base_polyhedron

    @property
    def center(self):
        return None
