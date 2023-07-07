import numpy as np
import itertools
import matplotlib.pyplot as plt
from typing import List
from pydrake.all import (
    Variables,
    DecomposeAffineExpressions,
    HPolyhedron,
    Formula,
    FormulaKind,
)

from large_gcs.contact.contact_pair_mode import (
    ContactPairMode,
    generate_contact_pair_modes,
)
from large_gcs.contact.rigid_body import RigidBody
from large_gcs.geometry.convex_set import ConvexSet


class ContactSet(ConvexSet):
    def __init__(
        self, contact_pair_modes: List[ContactPairMode], all_variables: List[Variables]
    ):
        self.contact_pair_modes = contact_pair_modes
        self.constraint_formulas = [
            constraint.item()
            for mode in contact_pair_modes
            for constraint in mode.constraint_formulas
        ]
        self._polyhedron = self._construct_polyhedron_from_constraints(
            self.constraint_formulas, all_variables
        )

    def _construct_polyhedron_from_constraints(
        self, constraints: List[Formula], variables: List[Variables]
    ):
        """
        Construct a polyhedron from a list of constraint formulas.
        Args:
            constraints: array of constraint formulas.
            variables: array of variables.
        """

        expressions = []
        for formula in constraints:
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

    @staticmethod
    def flatten_set_vars(vars_pos):
        return vars_pos.flatten()

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
