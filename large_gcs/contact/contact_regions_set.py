from dataclasses import dataclass
from typing import List

from pydrake.all import le

from large_gcs.contact.contact_location import ContactLocationVertex
from large_gcs.contact.contact_set_decision_variables import ContactSetDecisionVariables
from large_gcs.contact.rigid_body import RigidBody
from large_gcs.geometry.convex_set import ConvexSet
from large_gcs.geometry.geometry_utils import (
    HPolyhedronFromConstraints,
    scalar_proj_u_onto_v,
)
from large_gcs.geometry.polyhedron import Polyhedron


@dataclass
class ContactRegionParams:
    region_vertices: List
    obj_indices: List[int] = None
    rob_indices: List[int] = None


class ContactRegionsSet(ConvexSet):
    """A set where the position of each body is specified to be within a
    polyhedral region."""

    def __init__(
        self,
        objects: List[RigidBody],
        robots: List[RigidBody],
        contact_region_params: List[ContactRegionParams],
        name: str,
    ):
        self.contact_region_params = contact_region_params
        self.vars = ContactSetDecisionVariables.base_vars_from_objs_robs(
            objects, robots
        )

        self.constraint_formulas = []
        bodies_with_regions = set()

        def add_constraint_formulas(region: Polyhedron, body: RigidBody):
            if body.name in bodies_with_regions:
                raise ValueError(f"Body {body.name} is in multiple contact regions.")
            self.constraint_formulas.extend(
                self._generate_body_in_region_constraints(region, body)
            )
            bodies_with_regions.add(body.name)

        for region_params in contact_region_params:
            region = Polyhedron.from_vertices(region_params.region_vertices)
            if region_params.obj_indices is not None:
                for obj_index in region_params.obj_indices:
                    add_constraint_formulas(region, objects[obj_index])
            if region_params.rob_indices is not None:
                for rob_index in region_params.rob_indices:
                    add_constraint_formulas(region, robots[rob_index])

        self._polyhedron = Polyhedron.from_constraints(
            self.constraint_formulas, self.vars.base_all
        )
        self._base_polyhedron = HPolyhedronFromConstraints(
            self.constraint_formulas,
            self.vars.base_all,
            make_bounded=False,
        )
        self._name = name

    @staticmethod
    def _generate_body_in_region_constraints(region: Polyhedron, body: RigidBody):
        offset_b = region.set.b()
        for i, normal in enumerate(region.set.A()):
            projections = []
            for j in range(len(body.geometry.vertices)):
                contact_location = ContactLocationVertex(body, j)
                projections.append(scalar_proj_u_onto_v(contact_location.p_CV, normal))
            max_projection = max(projections)
            assert max_projection >= 0
            offset_b[i] -= max_projection
        return le(region.set.A() @ body.vars_base_pos, offset_b)

    @property
    def id(self):
        return self._name

    @property
    def dim(self):
        return self.set.A().shape[1]

    @property
    def set(self):
        return self._base_polyhedron

    @property
    def base_set(self):
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
