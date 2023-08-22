from dataclasses import dataclass
from typing import List

import numpy as np
from pydrake.all import Formula
from pydrake.all import Point as DrakePoint

from large_gcs.contact.contact_pair_mode import ContactPairMode, InContactPairMode
from large_gcs.contact.rigid_body import MobilityType, RigidBody


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
