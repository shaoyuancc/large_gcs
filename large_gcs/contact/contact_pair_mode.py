from dataclasses import dataclass
from abc import ABC, abstractmethod
from large_gcs.geometry.convex_set import ConvexSet
import numpy as np
import matplotlib.pyplot as plt
import itertools
from large_gcs.geometry.geometry_utils import *
from pydrake.all import MakeMatrixContinuousVariable, Expression, Formula
from large_gcs.contact.rigid_body import RigidBody, MobilityType
from large_gcs.contact.contact_location import (
    ContactLocation,
    ContactLocationVertex,
    ContactLocationFace,
    is_possible_face_face_contact,
    is_possible_face_vertex_contact,
)


@dataclass
class ContactPairMode(ABC):
    """Contact mode between two contact locations on two rigid bodies"""

    body_a: RigidBody
    body_b: RigidBody
    contact_location_a: ContactLocation
    contact_location_b: ContactLocation

    def __post_init__(self):
        assert self.body_a.dim == self.body_b.dim
        assert not (
            isinstance(self.contact_location_a, ContactLocationVertex)
            and isinstance(self.contact_location_b, ContactLocationVertex)
        ), "Vertex-vertex contact not supported"
        assert not (
            self.body_a.mobility_type == MobilityType.STATIC
            and self.body_b.mobility_type == MobilityType.STATIC
        ), "Static-static contact does not need to be considered"

    def plot(self, **kwargs):
        self._plot(**kwargs)
        plt.show()

    # @abstractmethod
    def to_convex_set(self) -> ConvexSet:
        pass


@dataclass
class NoContactPairMode(ContactPairMode):
    def __post_init__(self):
        super().__post_init__()
        # The halfspace boundary of body_a that contact location b is not within
        assert isinstance(self.contact_location_a, ContactLocationFace)

    def _plot(self, **kwargs):
        self.contact_location_a.plot(color="blue", **kwargs)
        self.contact_location_b.plot(color="blue", **kwargs)

    def to_convex_set(self) -> ConvexSet:

        # Position Constraints
        # Face-face no contact
        if isinstance(self.contact_location_b, ContactLocationFace):
            if self.body_a.mobility_type == MobilityType.STATIC:
                pass
            else:
                pass

        # Face-vertex no contact


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

    def to_convex_set(self) -> ConvexSet:
        """Generate appropriate constraint formulas and convert to convex set representation
        Cases that are handled:
            Face-face contact
                Static obstacle - unactuated object
                Static obstacle - actuated robot
                Unactuated object - actuated robot
            Face-vertex contact
                Static obstacle - unactuated object
                Static obstacle - actuated robot
                Unactuated object - actuated robot
        """

        # Face-face contact

        # Static obstacle - unactuated object
        # Static obstacle - actuated robot
        # Unactuated object - actuated robot

        # Face-vertex contact

        # Static obstacle - unactuated object
        # Static obstacle - actuated robot
        # Unactuated object - actuated robot
        pass


def generate_contact_pair_modes(body_a: RigidBody, body_b: RigidBody):
    """Generate all possible contact pair modes between two rigid bodies"""
    contact_pair_modes = []

    # Face-face contact
    for index_a, index_b in itertools.product(
        range(body_a.n_faces), range(body_b.n_faces)
    ):
        # Check if normals are in opposite directions
        face_a = ContactLocationFace(body=body_a, halfspace_index=index_a)
        face_b = ContactLocationFace(body=body_b, halfspace_index=index_b)
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
        face_a = ContactLocationFace(body=body_a, halfspace_index=index_a)
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
        face_b = ContactLocationFace(body=body_b, halfspace_index=index_b)
        if is_possible_face_vertex_contact(face_b, vertex_a):
            contact_pair_modes.append(
                InContactPairMode(
                    body_a=body_a,
                    body_b=body_b,
                    contact_location_a=vertex_a,
                    contact_location_b=face_b,
                )
            )

    # No contact relative to body_a
    no_contact_pair_modes = []
    for index_a in range(body_a.n_faces):
        in_contact_pair = next(
            filter(
                lambda x: isinstance(x.contact_location_a, ContactLocationFace)
                and x.contact_location_a.halfspace_index == index_a,
                contact_pair_modes,
            )
        )
        no_contact_pair_modes.append(in_contact_pair.to_no_contact_pair_mode())

    return contact_pair_modes + no_contact_pair_modes
