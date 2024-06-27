from dataclasses import dataclass
from typing import List

import numpy as np

from large_gcs.contact.contact_pair_mode import ContactPairMode, InContactPairMode
from large_gcs.contact.rigid_body import RigidBody


@dataclass
class ContactSetDecisionVariables:
    pos: np.ndarray
    force_act: np.ndarray
    force_mag_AB: np.ndarray
    all: np.ndarray
    base_all: np.ndarray
    n_objects: int
    n_robots: int

    @classmethod
    def from_contact_pair_modes(
        cls,
        objects: List[RigidBody],
        robots: List[RigidBody],
        contact_pair_modes: List[ContactPairMode],
    ):
        pos = np.array([body.vars_pos for body in objects + robots])

        force_act = np.array([body.vars_force_act for body in robots])
        in_contact_pair_modes = [
            mode for mode in contact_pair_modes if isinstance(mode, InContactPairMode)
        ]
        force_mag_AB = np.array(
            [mode.vars_force_mag_AB for mode in in_contact_pair_modes]
        )

        # All the decision variables for a single vertex
        all = np.concatenate(
            (
                pos.flatten(),
                force_act.flatten(),
                force_mag_AB.flatten(),
            )
        )
        # Extract the first point in n_pos_points_per_set
        base_all = np.array([body.vars_base_pos for body in objects + robots]).flatten()

        return cls(
            pos=pos,
            force_act=force_act,
            force_mag_AB=force_mag_AB,
            all=all,
            base_all=base_all,
            n_objects=len(objects),
            n_robots=len(robots),
        )

    @classmethod
    def base_vars_from_objs_robs(cls, objects, robots):
        """Base vars have no force variables, and only one knot point per
        set."""
        pos = np.array([body.vars_base_pos for body in objects + robots])
        pos = pos[:, :, np.newaxis]
        empty = np.array([])
        return cls(
            pos=pos,
            force_act=empty,
            force_mag_AB=empty,
            all=pos.flatten(),
            base_all=pos.flatten(),
            n_objects=len(objects),
            n_robots=len(robots),
        )

    def pos_from_all(self, vars_all):
        """Extracts the vars_pos from vars_all and reshapes it to match the
        template."""
        return np.reshape(vars_all[: self.pos.size], self.pos.shape)

    def last_pos_from_all(self, vars_all):
        """Extracts the last knot point vars_pos from vars_all."""
        return self.pos_from_all(vars_all)[:, :, -1].flatten()

    def first_pos_from_all(self, vars_all):
        """Extracts the first knot point vars_pos from vars_all."""
        return self.pos_from_all(vars_all)[:, :, 0].flatten()

    def obj_pos_from_all(self, vars_all):
        """Extracts the vars_pos for the objects from vars_all."""
        return self.pos_from_all(vars_all)[: self.n_objects]

    def rob_pos_from_all(self, vars_all):
        """Extracts the vars_pos for the robots from vars_all."""
        return self.pos_from_all(vars_all)[self.n_objects :]

    def obj_last_pos_from_all(self, vars_all):
        """Extracts the last knot point vars_pos for the objects from
        vars_all."""
        return self.obj_pos_from_all(vars_all)[:, :, -1].flatten()

    def rob_last_pos_from_all(self, vars_all):
        """Extracts the last knot point vars_pos for the robots from
        vars_all."""
        return self.rob_pos_from_all(vars_all)[:, :, -1].flatten()

    def obj_first_pos_from_all(self, vars_all):
        """Extracts the first knot point vars_pos for the objects from
        vars_all."""
        return self.obj_pos_from_all(vars_all)[:, :, 0].flatten()

    def rob_first_pos_from_all(self, vars_all):
        """Extracts the first knot point vars_pos for the robots from
        vars_all."""
        return self.rob_pos_from_all(vars_all)[:, :, 0].flatten()

    @property
    def last_pos(self):
        return self.pos[:, :, -1].flatten()

    @property
    def first_pos(self):
        return self.pos[:, :, 0].flatten()

    @property
    def obj_pos(self):
        return self.pos[: self.n_objects]

    @property
    def rob_pos(self):
        return self.pos[self.n_objects :]

    @property
    def obj_last_pos(self):
        return self.obj_pos[:, :, -1].flatten()

    @property
    def rob_last_pos(self):
        return self.rob_pos[:, :, -1].flatten()

    @property
    def obj_first_pos(self):
        return self.obj_pos[:, :, 0].flatten()

    @property
    def rob_first_pos(self):
        return self.rob_pos[:, :, 0].flatten()
