import numpy as np
from dataclasses import dataclass
import itertools
import matplotlib.pyplot as plt
from typing import List
from pydrake.all import (
    Variables,
    DecomposeAffineExpressions,
    HPolyhedron,
    Formula,
    FormulaKind,
    le,
)

from large_gcs.contact.contact_pair_mode import (
    ContactPairMode,
    InContactPairMode,
)
from large_gcs.contact.rigid_body import RigidBody
from large_gcs.geometry.convex_set import ConvexSet


@dataclass
class ContactSetDecisionVariables:
    pos: np.ndarray
    force_res: np.ndarray
    force_act: np.ndarray
    force_mag_AB: np.ndarray
    force_mag_BA: np.ndarray
    all: np.ndarray

    def __init__(
        self,
        objects: List[RigidBody],
        robots: List[RigidBody],
        in_contact_pair_modes: List[InContactPairMode],
    ):
        self.pos = np.array([body.vars_pos for body in objects + robots])
        self.force_res = np.array([body.vars_force_res for body in objects + robots])

        self.force_act = np.array([body.vars_force_act for body in robots])
        self.force_mag_AB = np.array(
            [mode.vars_force_mag_AB for mode in in_contact_pair_modes]
        )
        self.force_mag_BA = np.array(
            [mode.vars_force_mag_BA for mode in in_contact_pair_modes]
        )

        # All the decision variables for a single vertex
        self.all = np.concatenate(
            (
                self.pos.flatten(),
                self.force_res.flatten(),
                self.force_act.flatten(),
                self.force_mag_AB.flatten(),
                self.force_mag_BA.flatten(),
            )
        )

    def pos_from_all(self, vars_all):
        """Extracts the vars_pos from vars_all and reshapes it to match the template"""
        return np.reshape(vars_all[: self.pos.size], self.pos.shape)

    def force_res_from_vars(self, vars_all):
        return np.reshape(
            vars_all[self.pos.size : self.pos.size + self.force_res.size],
            self.force_res.shape,
        )


class ContactSet(ConvexSet):
    def __init__(
        self,
        contact_pair_modes: List[ContactPairMode],
        set_force_constraints: List[Formula],
        all_variables: np.ndarray,
    ):
        # print(f"set_force_constraints shape: {np.array(set_force_constraints).shape}")
        # print(f"set_force_constraints: {set_force_constraints}")

        self.contact_pair_modes = contact_pair_modes
        self.constraint_formulas = [
            constraint.item()
            for mode in contact_pair_modes
            for constraint in mode.constraint_formulas
        ]
        self.constraint_formulas.extend(set_force_constraints)
        self._polyhedron = self._construct_polyhedron_from_constraints(
            self.constraint_formulas, all_variables
        )
        # print(f"set id: {self.id}")
        # print(f"{all_variables}")
        # print()

    def _construct_polyhedron_from_constraints(
        self,
        constraints: List[Formula],
        variables: np.ndarray,
        make_bounded: bool = True,
        BOUND: float = 1000.0,
    ):
        """
        Construct a polyhedron from a list of constraint formulas.
        Args:
            constraints: array of constraint formulas.
            variables: array of variables.
        """
        if make_bounded:
            ub = np.ones(variables.shape) * BOUND
            upper_limits = le(variables, ub)
            lower_limits = le(-ub, variables)
            limits = np.concatenate((upper_limits, lower_limits))
            constraints = np.append(constraints, limits)

        expressions = []
        # print(f"Constructing polyhedron for set {self.id}")
        for formula in constraints:
            # print(formula)
            kind = formula.get_kind()
            lhs, rhs = formula.Unapply()[1]
            if kind == FormulaKind.Eq:
                # Eq constraint ax = b is
                # implemented as ax ≤ b, -ax <= -b
                expressions.append(lhs - rhs)
                expressions.append(rhs - lhs)
            elif kind == FormulaKind.Geq:
                # lhs >= rhs
                # ==> rhs - lhs ≤ 0
                expressions.append(rhs - lhs)
            elif kind == FormulaKind.Leq:
                # lhs ≤ rhs
                # ==> lhs - rhs ≤ 0
                expressions.append(lhs - rhs)
            else:
                raise NotImplementedError("Type of constraint formula not implemented")

        # We now have expr ≤ 0 for all expressions
        # ==> we get Ax - b ≤ 0
        A, b_neg = DecomposeAffineExpressions(expressions, variables)

        # Polyhedrons are of the form: Ax <= b
        b = -b_neg
        polyhedron = HPolyhedron(A, b)

        return polyhedron

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
    def center(self):
        return None
