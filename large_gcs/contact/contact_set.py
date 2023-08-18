from dataclasses import dataclass
from typing import List

import numpy as np
from pydrake.all import Formula
from pydrake.all import Point as DrakePoint

from large_gcs.contact.contact_pair_mode import ContactPairMode, InContactPairMode
from large_gcs.contact.rigid_body import MobilityType, RigidBody
from large_gcs.geometry.convex_set import ConvexSet
from large_gcs.geometry.geometry_utils import HPolyhedronFromConstraints


@dataclass
class ContactSetDecisionVariables:
    pos: np.ndarray
    force_res: np.ndarray
    force_act: np.ndarray
    force_mag_AB: np.ndarray
    force_mag_BA: np.ndarray
    all: np.ndarray
    base_all: np.ndarray

    @classmethod
    def from_factored_collision_free_body(cls, body: RigidBody):
        empty = np.array([])
        pos = body.vars_pos[np.newaxis, :]
        return cls(
            pos=pos,
            force_res=empty,
            force_act=empty,
            force_mag_AB=empty,
            force_mag_BA=empty,
            all=pos.flatten(),
            base_all=pos[:, 0].flatten(),
        )

    @classmethod
    def from_contact_pair_modes(
        cls,
        objects: List[RigidBody],
        robots: List[RigidBody],
        contact_pair_modes: List[ContactPairMode],
    ):
        pos = np.array([body.vars_pos for body in objects + robots])
        force_res = np.array([body.vars_force_res for body in objects + robots])

        force_act = np.array([body.vars_force_act for body in robots])
        in_contact_pair_modes = [
            mode for mode in contact_pair_modes if isinstance(mode, InContactPairMode)
        ]
        force_mag_AB = np.array(
            [mode.vars_force_mag_AB for mode in in_contact_pair_modes]
        )
        force_mag_BA = np.array(
            [mode.vars_force_mag_BA for mode in in_contact_pair_modes]
        )

        # All the decision variables for a single vertex
        all = np.concatenate(
            (
                pos.flatten(),
                force_res.flatten(),
                force_act.flatten(),
                force_mag_AB.flatten(),
                force_mag_BA.flatten(),
            )
        )
        # Extract the first point in n_pos_points_per_set
        base_all = np.array([body.vars_base_pos for body in objects + robots]).flatten()

        return cls(pos, force_res, force_act, force_mag_AB, force_mag_BA, all, base_all)

    def pos_from_all(self, vars_all):
        """Extracts the vars_pos from vars_all and reshapes it to match the template"""
        return np.reshape(vars_all[: self.pos.size], self.pos.shape)

    def force_res_from_vars(self, vars_all):
        return np.reshape(
            vars_all[self.pos.size : self.pos.size + self.force_res.size],
            self.force_res.shape,
        )

    @classmethod
    def from_objs_robs(cls, objects, robots):
        pos = np.array([body.vars_base_pos for body in objects + robots])
        pos = pos[:, :, np.newaxis]
        empty = np.array([])
        return cls(pos, empty, empty, empty, empty, pos.flatten(), pos.flatten())


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
            constraint.item()
            for mode in contact_pair_modes
            for constraint in mode.constraint_formulas
        ]
        self.base_constraint_formulas = [
            constraint.item()
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
