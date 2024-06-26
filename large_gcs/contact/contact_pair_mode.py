import itertools
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import Expression, Formula, HPolyhedron, Variable, eq, ge, le

from large_gcs.contact.contact_location import (
    ContactLocation,
    ContactLocationFace,
    ContactLocationVertex,
    is_possible_face_face_contact,
    is_possible_face_vertex_contact,
)
from large_gcs.contact.rigid_body import MobilityType, RigidBody
from large_gcs.geometry.geometry_utils import HPolyhedronFromConstraints


@dataclass
class ContactPairMode(ABC):
    """Contact mode between two contact locations on two rigid bodies."""

    body_a: RigidBody
    body_b: RigidBody
    contact_location_a: Optional[ContactLocation]
    contact_location_b: Optional[ContactLocation]

    def __post_init__(self):
        assert self.body_a.dim == self.body_b.dim
        if self.contact_location_a is not None and self.contact_location_b is not None:
            assert not (
                isinstance(self.contact_location_a, ContactLocationVertex)
                and isinstance(self.contact_location_b, ContactLocationVertex)
            ), "Vertex-vertex contact not supported"
            assert not (
                self.body_a.mobility_type == MobilityType.STATIC
                and self.body_b.mobility_type == MobilityType.STATIC
            ), "Static-static contact does not need to be considered"
        self._create_decision_vars()
        (
            self.constraint_formulas,
            self.base_constraint_formulas,
        ) = self._create_constraint_formulas()

    def plot(self, **kwargs):
        plt.title(self.id, y=1.05)
        self._plot(**kwargs)
        plt.show()

    @abstractmethod
    def _create_constraint_formulas(self) -> Tuple[List[Formula], List[Formula]]:
        pass

    def _create_decision_vars(self):
        pass

    def _create_signed_dist_surrog_constraint_exprs(self) -> List[Expression]:
        exprs = []
        if self.body_a.mobility_type == MobilityType.STATIC:
            if isinstance(self.contact_location_a, ContactLocationFace):
                if isinstance(self.contact_location_b, ContactLocationFace):
                    exprs = create_static_face_movable_face_signed_dist_surrog_exprs(
                        self.contact_location_a, self.contact_location_b
                    )
                elif isinstance(self.contact_location_b, ContactLocationVertex):
                    exprs = create_static_face_movable_vert_signed_dist_surrog_exprs(
                        self.contact_location_a, self.contact_location_b
                    )
            elif isinstance(self.contact_location_a, ContactLocationVertex):
                if isinstance(self.contact_location_b, ContactLocationFace):
                    exprs = create_static_vert_movable_face_signed_dist_surrog_exprs(
                        self.contact_location_a, self.contact_location_b
                    )
        else:  # body_a is movable
            if isinstance(self.contact_location_a, ContactLocationFace):
                if isinstance(self.contact_location_b, ContactLocationFace):
                    exprs = create_movable_face_face_signed_dist_surrog_exprs(
                        self.contact_location_a, self.contact_location_b
                    )
                elif isinstance(self.contact_location_b, ContactLocationVertex):
                    exprs = create_movable_face_vert_signed_dist_surrog_exprs(
                        self.contact_location_a, self.contact_location_b
                    )
            elif isinstance(self.contact_location_a, ContactLocationVertex):
                if isinstance(self.contact_location_b, ContactLocationFace):
                    exprs = create_movable_face_vert_signed_dist_surrog_exprs(
                        self.contact_location_b, self.contact_location_a
                    )
                    # exprs = [
                    #     -expr
                    #     for expr in create_movable_face_vert_signed_dist_surrog_exprs(
                    #         self.contact_location_b, self.contact_location_a
                    #     )
                    # ]
        assert len(exprs) > 0
        return exprs

    @property
    def id(self) -> str:
        return f"{self.compact_class_name}|{self.body_a.name}_{self.contact_location_a.compact_name}-{self.body_b.name}_{self.contact_location_b.compact_name}"

    @property
    @abstractmethod
    def compact_class_name(self):
        pass

    @property
    def body_pair(self):
        return (self.body_a.name, self.body_b.name)

    @property
    def params(self):
        return ContactPairModeParams(
            type=type(self),
            body_a_name=self.body_a.name,
            body_b_name=self.body_b.name,
            contact_location_a_type=type(self.contact_location_a),
            contact_location_b_type=type(self.contact_location_b),
            contact_location_a_index=self.contact_location_a.index,
            contact_location_b_index=self.contact_location_b.index,
        )

    # Note: Didn't type hint vars because it led to a cyclic import, TODO: Figure out how to do this properly
    def create_base_polyhedron(
        self, vars, additional_constraints: List[Formula] = None
    ) -> HPolyhedron:
        """Args:
        vars: ContactSetDecisionVariables
        additional_constraints: Additional constraints to add to the base polyhedron
        """
        constraints = copy(self.base_constraint_formulas)
        if additional_constraints is not None:
            constraints += additional_constraints
        return HPolyhedronFromConstraints(
            constraints,
            vars.base_all,
            make_bounded=True,  # Should be bounded by workspace constraints
        )


@dataclass
class ContactPairModeParams:
    type: Type[ContactPairMode]
    body_a_name: str
    body_b_name: str
    contact_location_a_type: Type[ContactLocation]
    contact_location_b_type: Type[ContactLocation]
    contact_location_a_index: int
    contact_location_b_index: int


@dataclass
class NoContactPairMode(ContactPairMode):
    def __post_init__(self):
        super().__post_init__()

    def _plot(self, **kwargs):
        self.contact_location_a.plot(color="blue", **kwargs)
        self.contact_location_b.plot(color="blue", **kwargs)

    def _create_constraint_formulas(self):
        constraints = []
        base_constraints = []
        # Position Constraints
        signed_dist_exprs = self._create_signed_dist_surrog_constraint_exprs()
        constraints += [ge(expr, 0).item() for expr in signed_dist_exprs]
        # Extract the base signed distance constraint which is the one that involves the base position variables
        # i.e. pos 0 of n_pos_per_set. For sd, each pos produces one expression
        base_constraints = [constraints[0]]

        return constraints, base_constraints

    @property
    def compact_class_name(self):
        return "NC"


@dataclass
class RelaxedContactPairMode(ContactPairMode):
    contact_location_a: Optional[ContactLocation] = None
    contact_location_b: Optional[ContactLocation] = None

    def __post_init__(self):
        assert (
            self.contact_location_a is None and self.contact_location_b is None
        ), "Relaxed contact pair modes must not specify contact locations"
        super().__post_init__()

    def _plot(self, **kwargs):
        self.contact_location_a.plot(**kwargs)
        self.contact_location_b.plot(**kwargs)

    @property
    def id(self) -> str:
        return f"{self.compact_class_name}|{self.body_a.name}-{self.body_b.name}"


@dataclass
class RelaxedInContactPairMode(RelaxedContactPairMode):
    """Note: RelaxedInContactPairMode generated wrt the faces in body A (not body B)"""

    def _create_constraint_formulas(self):
        constraints = []
        base_constraints = []

        no_contact_pair_modes = generate_no_contact_pair_modes(self.body_a, self.body_b)
        # Position Constraints
        for nc_mode in no_contact_pair_modes:
            signed_dist_exprs = nc_mode._create_signed_dist_surrog_constraint_exprs()
            # Note how the position constraint is flipped to be <= 0 (compared to the NC case)
            pos_constraints = [le(expr, 0).item() for expr in signed_dist_exprs]
            constraints += pos_constraints
            base_constraints += [pos_constraints[0]]

        # Velocity Constraints
        # Make all velocities equal
        vel_constraints = []
        for body in [self.body_a, self.body_b]:
            if body.mobility_type == MobilityType.ACTUATED:
                vel_constraints += eq(body.vars_vel, body.vars_vel).tolist()
        vel_constraints += eq(self.body_a.vars_vel, self.body_b.vars_vel).tolist()

        constraints += vel_constraints

        return constraints, base_constraints

    @property
    def compact_class_name(self):
        return "RIC"


@dataclass
class RelaxedNoContactPairMode(RelaxedContactPairMode):
    """RelaxedNoContactPairMode has no constraints, functions kind of like a
    placeholder in the contact set."""

    def _create_constraint_formulas(self):
        """No constraints."""
        constraints = []
        base_constraints = []

        return constraints, base_constraints

    @property
    def compact_class_name(self):
        return "RNC"


@dataclass
class InContactPairMode(ContactPairMode):
    def to_no_contact_pair_mode(self) -> NoContactPairMode:
        return NoContactPairMode(
            body_a=self.body_a,
            body_b=self.body_b,
            contact_location_a=self.contact_location_a,
            contact_location_b=self.contact_location_b,
        )

    def _plot(self, **kwargs):
        self.contact_location_a.plot(**kwargs)
        self.contact_location_b.plot(**kwargs)

    def _create_decision_vars(self):
        # Magnitude of the forces from body_a to body_b and vice versa
        self.vars_force_mag_AB = Variable(
            f"{self.id}_force_mag_AB", type=Variable.Type.CONTINUOUS
        )
        # By Newton's Third Law, magnitude of forces A on B and B on A are equal
        self.vars_force_mag_BA = self.vars_force_mag_AB

    def _create_constraint_formulas(self):
        constraints = []
        base_constraints = []
        # Position Constraints
        sd_exprs = self._create_signed_dist_surrog_constraint_exprs()
        sd_constraints = [eq(expr, 0).item() for expr in sd_exprs]
        constraints += sd_constraints
        # Extract the base signed distance constraint which is the one that involves the base position variables
        # i.e. pos 0 of n_pos_per_set. For sd, each pos produces one expression
        base_constraints += [sd_constraints[0]]

        hb_constraints = self._create_horizontal_bounds_formulas()
        constraints += hb_constraints
        # Extract the base horizontal bound constraint which is the one that involves the base position variables
        # i.e. pos 0 of n_pos_per_set. For hb, each pos produces two formulas (lower and upper bound)
        base_constraints += [hb_constraints[0], hb_constraints[1]]

        # Force Constraints
        constraints += self._create_force_constraint_formulas()

        return constraints, base_constraints

    def _create_force_constraint_formulas(self):
        formulas = []
        # If bodies A and B are in contact, A must be exerting some positive force on B, and vice versa
        formulas.append(ge(self.vars_force_mag_AB, 0).item())
        return formulas

    def _create_horizontal_bounds_formulas(self):
        formulas = []
        if self.body_a.mobility_type == MobilityType.STATIC:
            if isinstance(self.contact_location_a, ContactLocationFace):
                if isinstance(self.contact_location_b, ContactLocationFace):
                    formulas = (
                        create_static_face_movable_face_horizontal_bounds_formulas(
                            self.contact_location_a, self.contact_location_b
                        )
                    )
                elif isinstance(self.contact_location_b, ContactLocationVertex):
                    formulas = (
                        create_static_face_movable_vert_horizontal_bounds_formulas(
                            self.contact_location_a, self.contact_location_b
                        )
                    )
            elif isinstance(self.contact_location_a, ContactLocationVertex):
                if isinstance(self.contact_location_b, ContactLocationFace):
                    formulas = (
                        create_static_vert_movable_face_horizontal_bounds_formulas(
                            self.contact_location_a, self.contact_location_b
                        )
                    )
        else:  # body_a is movable
            if isinstance(self.contact_location_a, ContactLocationFace):
                if isinstance(self.contact_location_b, ContactLocationFace):
                    formulas = create_movable_face_face_horizontal_bounds_formulas(
                        self.contact_location_a, self.contact_location_b
                    )
                elif isinstance(self.contact_location_b, ContactLocationVertex):
                    formulas = create_movable_face_vert_horizontal_bounds_formulas(
                        self.contact_location_a, self.contact_location_b
                    )
            elif isinstance(self.contact_location_a, ContactLocationVertex):
                if isinstance(self.contact_location_b, ContactLocationFace):
                    formulas = create_movable_face_vert_horizontal_bounds_formulas(
                        face=self.contact_location_b, vert=self.contact_location_a
                    )
        assert len(formulas) > 0
        return formulas

    @property
    def compact_class_name(self):
        return "IC"

    @property
    def unit_normal(self):
        """Unit contact normal from body_a to body_b."""
        if isinstance(self.contact_location_a, ContactLocationFace):
            return self.contact_location_a.unit_normal
        elif isinstance(self.contact_location_a, ContactLocationVertex) and isinstance(
            self.contact_location_b, ContactLocationFace
        ):
            return -self.contact_location_b.unit_normal


def create_static_face_movable_face_signed_dist_surrog_exprs(
    static_face: ContactLocationFace, movable_face: ContactLocationFace
):
    """Create an expression for a surrogate of the signed distance between a
    static face and a movable face."""
    assert static_face.body.mobility_type == MobilityType.STATIC
    assert movable_face.body.mobility_type != MobilityType.STATIC
    assert static_face.body.dim == movable_face.body.dim
    assert np.allclose(
        static_face.unit_normal, -movable_face.unit_normal
    ), "Only valid for opposing faces"

    exprs = []
    for p_Mc in movable_face.body.vars_pos.T:
        dist_surrog = _plane_to_point_dist_surrog_exprs(
            normal=static_face.normal,
            p_plane_point=static_face.adj_vertices[0],
            p_target_point=p_Mc + movable_face.p_CF,
        )
        exprs.append(dist_surrog)
    return exprs


def create_static_face_movable_vert_signed_dist_surrog_exprs(
    static_face: ContactLocationFace, movable_vertex: ContactLocationVertex
):
    assert static_face.body.mobility_type == MobilityType.STATIC
    assert movable_vertex.body.mobility_type != MobilityType.STATIC
    assert static_face.body.dim == movable_vertex.body.dim

    exprs = []
    for p_Mc in movable_vertex.body.vars_pos.T:
        dist_surrog = _plane_to_point_dist_surrog_exprs(
            normal=static_face.normal,
            p_plane_point=static_face.adj_vertices[0],
            p_target_point=p_Mc + movable_vertex.p_CV,
        )
        exprs.append(dist_surrog)
    return exprs


def create_static_vert_movable_face_signed_dist_surrog_exprs(
    static_vert: ContactLocationVertex, movable_face: ContactLocationFace
):
    assert static_vert.body.mobility_type == MobilityType.STATIC
    assert movable_face.body.mobility_type != MobilityType.STATIC
    assert static_vert.body.dim == movable_face.body.dim

    exprs = []
    for p_Mc in movable_face.body.vars_pos.T:
        dist_surrog = _plane_to_point_dist_surrog_exprs(
            normal=movable_face.normal,
            p_plane_point=p_Mc + movable_face.p_CF,
            p_target_point=static_vert.vertex,
        )
        exprs.append(dist_surrog)
    return exprs


def create_movable_face_face_signed_dist_surrog_exprs(
    face_a: ContactLocationFace, face_b: ContactLocationFace
):
    assert face_a.body.mobility_type != MobilityType.STATIC
    assert face_b.body.mobility_type != MobilityType.STATIC
    assert face_a.body.dim == face_b.body.dim
    assert np.allclose(
        face_a.unit_normal, -face_b.unit_normal
    ), "Only valid for opposing faces"

    exprs = []
    # p_Mca is position of (M)ovable (c)enter of body_(a)
    for p_Mca, p_Mcb in zip(face_a.body.vars_pos.T, face_b.body.vars_pos.T):
        dist_surrog = _plane_to_point_dist_surrog_exprs(
            normal=face_a.normal,
            p_plane_point=p_Mca + face_a.p_CF,
            p_target_point=p_Mcb + face_b.p_CF,
        )
        exprs.append(dist_surrog)
    return exprs


def create_movable_face_vert_signed_dist_surrog_exprs(
    face_a: ContactLocationFace, vert_b: ContactLocationVertex
):
    assert face_a.body.mobility_type != MobilityType.STATIC
    assert vert_b.body.mobility_type != MobilityType.STATIC
    assert face_a.body.dim == vert_b.body.dim

    exprs = []
    # p_Mca is position of (M)ovable (c)enter of body_(a)
    for p_Mca, p_Mcb in zip(face_a.body.vars_pos.T, vert_b.body.vars_pos.T):
        dist_surrog = _plane_to_point_dist_surrog_exprs(
            normal=face_a.normal,
            p_plane_point=p_Mca + face_a.p_CF,
            p_target_point=p_Mcb + vert_b.p_CV,
        )
        exprs.append(dist_surrog)
    return exprs


def _plane_to_point_dist_surrog_exprs(normal, p_plane_point, p_target_point):
    """Point to plane formula surrogate."""
    p_PT = p_target_point - p_plane_point
    dist_surrog = np.dot(normal, p_PT) / np.dot(normal, normal)
    # The dist squared would be: (np.dot(normal, p_PT) ** 2) / np.dot(normal, normal)
    # The dist would be: np.dot(normal, p_PT) / sqrt(np.dot(normal, normal))
    # By using this expression, we get a polynomial expression (no square root), and we maintain the sign of the distance
    return dist_surrog


def create_static_face_movable_face_horizontal_bounds_formulas(
    static_face: ContactLocationFace, movable_face: ContactLocationFace
):
    assert static_face.body.mobility_type == MobilityType.STATIC
    assert movable_face.body.mobility_type != MobilityType.STATIC
    assert static_face.body.dim == movable_face.body.dim
    assert np.allclose(
        static_face.unit_normal, -movable_face.unit_normal
    ), "Only valid for opposing faces"

    formulas = []
    for p_Mc in movable_face.body.vars_pos.T:
        formulas += _face_horizontal_bounds_formulas(
            p_Refleft=static_face.adj_vertices[1],
            p_Refright=static_face.adj_vertices[0],
            p_Relv=p_Mc + movable_face.p_CVleft,
            rel_length=movable_face.length,
        )

    return formulas


def create_static_face_movable_vert_horizontal_bounds_formulas(
    static_face: ContactLocationFace, movable_vert: ContactLocationVertex
):
    assert static_face.body.mobility_type == MobilityType.STATIC
    assert movable_vert.body.mobility_type != MobilityType.STATIC
    assert static_face.body.dim == movable_vert.body.dim

    formulas = []
    for p_Mc in movable_vert.body.vars_pos.T:
        formulas += _face_horizontal_bounds_formulas(
            p_Refleft=static_face.adj_vertices[1],
            p_Refright=static_face.adj_vertices[0],
            p_Relv=p_Mc + movable_vert.p_CV,
            rel_length=0,
        )

    return formulas


def create_static_vert_movable_face_horizontal_bounds_formulas(
    static_vert: ContactLocationVertex, movable_face: ContactLocationFace
):
    assert static_vert.body.mobility_type == MobilityType.STATIC
    assert movable_face.body.mobility_type != MobilityType.STATIC
    assert static_vert.body.dim == movable_face.body.dim

    formulas = []
    for p_Mc in movable_face.body.vars_pos.T:
        formulas += _face_horizontal_bounds_formulas(
            p_Refleft=p_Mc + movable_face.p_CVleft,
            p_Refright=p_Mc + movable_face.p_CVright,
            p_Relv=static_vert.vertex,
            rel_length=0,
        )
    return formulas


def create_movable_face_face_horizontal_bounds_formulas(
    face_a: ContactLocationFace, face_b: ContactLocationFace
):
    assert face_a.body.mobility_type != MobilityType.STATIC
    assert face_b.body.mobility_type != MobilityType.STATIC
    assert face_a.body.dim == face_b.body.dim
    assert np.allclose(
        face_a.unit_normal, -face_b.unit_normal
    ), "Only valid for opposing faces"
    formulas = []
    for p_Mca, p_Mcb in zip(face_a.body.vars_pos.T, face_b.body.vars_pos.T):
        formulas += _face_horizontal_bounds_formulas(
            p_Refleft=p_Mca + face_a.p_CVleft,
            p_Refright=p_Mca + face_a.p_CVright,
            p_Relv=p_Mcb + face_b.p_CVleft,
            rel_length=face_b.length,
        )
    return formulas


def create_movable_face_vert_horizontal_bounds_formulas(
    face: ContactLocationFace, vert: ContactLocationVertex
):
    assert face.body.mobility_type != MobilityType.STATIC
    assert vert.body.mobility_type != MobilityType.STATIC
    assert face.body.dim == vert.body.dim

    formulas = []
    for p_Mca, p_Mcb in zip(face.body.vars_pos.T, vert.body.vars_pos.T):
        formulas += _face_horizontal_bounds_formulas(
            p_Refleft=p_Mca + face.p_CVleft,
            p_Refright=p_Mca + face.p_CVright,
            p_Relv=p_Mcb + vert.p_CV,
            rel_length=0,
        )
    return formulas


def _face_horizontal_bounds_formulas(
    p_Refleft, p_Refright, p_Relv, rel_length=0, buffer_ratio=0.0
):
    """Formulas for the horizontal bounds of a face-face contact such that the
    relative face is within the horizontal bounds (viewing the reference face
    normal as pointing upwards) of the reference face.

    Note that increasing the buffer_ratio will affect whether the sets intersect. If allowing sets to be revisited multiple
    times is not implemented, this will result in most problems becoming infeasible because you can't transition out of a contact
    mode since you need to go back to the non-contact mode but you already went through that to get to the contact mode in the
    first place.
    Args:
        p_Refleft (np.array): Position of the "left" vertex on the reference face (when viewing the reference face normal as pointing upwards)
        p_Refright (np.array): Position of the "right" vertex on the reference face
        p_Relv (np.array): Position of relative vertex (for face-face contact, should be the left vertex of the relative face)
        rel_length (float): Length of the relative face (if face-vertex contact then this should be 0)
    Returns:
        list: List of formulas
    """
    p_RefleftRefright = p_Refright - p_Refleft  # This should be a vector of numbers
    ref_length = np.linalg.norm(p_RefleftRefright)
    p_RefleftRelv_hat = p_RefleftRefright / ref_length
    p_RefleftRelv = p_Relv - p_Refleft
    # Project p_RefleftReleft onto p_RefleftRefright
    # Symbolic expression in decision variables
    dist = np.dot(p_RefleftRelv, p_RefleftRelv_hat)
    buff = buffer_ratio * ref_length
    lb = buff
    ub = ref_length + rel_length - buff
    return [ge(dist, lb).item(), le(dist, ub).item()]


def generate_contact_pair_modes(
    body_a: RigidBody, body_b: RigidBody, ignore_static_actuated_contact=True
):
    """Generate all possible contact pair modes between two rigid bodies."""
    # No contact relative to body_a
    no_contact_pair_modes = generate_no_contact_pair_modes(body_a=body_a, body_b=body_b)

    if ignore_static_actuated_contact and (
        (
            body_a.mobility_type == MobilityType.STATIC
            or body_b.mobility_type == MobilityType.STATIC
        )
        and (
            body_a.mobility_type == MobilityType.ACTUATED
            or body_b.mobility_type == MobilityType.ACTUATED
        )
    ):
        return no_contact_pair_modes
    else:
        in_contact_pair_modes = generate_in_contact_pair_modes(body_a, body_b)
        return in_contact_pair_modes + no_contact_pair_modes


def generate_relaxed_contact_pair_modes(body_a: RigidBody, body_b: RigidBody):
    return [
        RelaxedNoContactPairMode(body_a=body_a, body_b=body_b),
        RelaxedInContactPairMode(body_a=body_a, body_b=body_b),
    ]


def generate_cfree_contact_pair_modes(body_a: RigidBody, body_b: RigidBody):
    # No contact relative to body_a
    no_contact_pair_modes = generate_no_contact_pair_modes(body_a=body_a, body_b=body_b)

    in_contact_pair_modes = []
    # This is necessary because the body may be in contact and then transition to no contact
    # Vertex-face contact
    for index_a, index_b in itertools.product(
        range(body_a.n_vertices), range(body_b.n_faces)
    ):
        vertex_a = ContactLocationVertex(body=body_a, index=index_a)
        face_b = ContactLocationFace(body=body_b, index=index_b)
        if is_possible_face_vertex_contact(face_b, vertex_a):
            in_contact_pair_modes.append(
                InContactPairMode(
                    body_a=body_a,
                    body_b=body_b,
                    contact_location_a=vertex_a,
                    contact_location_b=face_b,
                )
            )

    return in_contact_pair_modes + no_contact_pair_modes


def generate_in_contact_pair_modes(body_a: RigidBody, body_b: RigidBody):
    contact_pair_modes = []
    # Face-face contact
    for index_a, index_b in itertools.product(
        range(body_a.n_faces), range(body_b.n_faces)
    ):
        # Check if normals are in opposite directions
        face_a = ContactLocationFace(body=body_a, index=index_a)
        face_b = ContactLocationFace(body=body_b, index=index_b)
        if is_possible_face_face_contact(face_a, face_b):
            contact_pair_modes.append(
                InContactPairMode(
                    body_a=body_a,
                    body_b=body_b,
                    contact_location_a=face_a,
                    contact_location_b=face_b,
                )
            )

    # Face-vertex contact
    for index_a, index_b in itertools.product(
        range(body_a.n_faces), range(body_b.n_vertices)
    ):
        face_a = ContactLocationFace(body=body_a, index=index_a)
        vertex_b = ContactLocationVertex(body=body_b, index=index_b)
        if is_possible_face_vertex_contact(face_a, vertex_b):
            contact_pair_modes.append(
                InContactPairMode(
                    body_a=body_a,
                    body_b=body_b,
                    contact_location_a=face_a,
                    contact_location_b=vertex_b,
                )
            )

    # Vertex-face contact
    for index_a, index_b in itertools.product(
        range(body_a.n_vertices), range(body_b.n_faces)
    ):
        vertex_a = ContactLocationVertex(body=body_a, index=index_a)
        face_b = ContactLocationFace(body=body_b, index=index_b)
        if is_possible_face_vertex_contact(face_b, vertex_a):
            contact_pair_modes.append(
                InContactPairMode(
                    body_a=body_a,
                    body_b=body_b,
                    contact_location_a=vertex_a,
                    contact_location_b=face_b,
                )
            )
    return contact_pair_modes


def generate_no_contact_pair_modes(body_a: RigidBody, body_b: RigidBody):
    no_contact_pair_modes = []
    # Face-face no contact
    for index_a, index_b in itertools.product(
        range(body_a.n_faces), range(body_b.n_faces)
    ):
        # Check if normals are in opposite directions
        face_a = ContactLocationFace(body=body_a, index=index_a)
        face_b = ContactLocationFace(body=body_b, index=index_b)
        if is_possible_face_face_contact(face_a, face_b):
            no_contact_pair_modes.append(
                NoContactPairMode(
                    body_a=body_a,
                    body_b=body_b,
                    contact_location_a=face_a,
                    contact_location_b=face_b,
                )
            )

    # Face-vertex no contact
    for index_a, index_b in itertools.product(
        range(body_a.n_faces), range(body_b.n_vertices)
    ):
        face_a = ContactLocationFace(body=body_a, index=index_a)
        vertex_b = ContactLocationVertex(body=body_b, index=index_b)
        if is_possible_face_vertex_contact(face_a, vertex_b):
            no_contact_pair_modes.append(
                NoContactPairMode(
                    body_a=body_a,
                    body_b=body_b,
                    contact_location_a=face_a,
                    contact_location_b=vertex_b,
                )
            )
    # Vertex-Face no contact
    for index_a, index_b in itertools.product(
        range(body_a.n_vertices), range(body_b.n_faces)
    ):
        vertex_a = ContactLocationVertex(body=body_a, index=index_a)
        face_b = ContactLocationFace(body=body_b, index=index_b)
        if is_possible_face_vertex_contact(face_b, vertex_a):
            no_contact_pair_modes.append(
                NoContactPairMode(
                    body_a=body_a,
                    body_b=body_b,
                    contact_location_a=vertex_a,
                    contact_location_b=face_b,
                )
            )
    return no_contact_pair_modes
