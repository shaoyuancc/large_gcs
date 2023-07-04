from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import itertools
from pydrake.all import MakeMatrixContinuousVariable, Expression, Formula
from large_gcs.geometry.geometry_utils import *
from large_gcs.contact.rigid_body import RigidBody


@dataclass
class ContactLocation:
    body: RigidBody

    def plot(self, **kwargs):
        if self.body.dim != 2:
            raise NotImplementedError
        self.body.geometry.plot()
        options = {"color": "r", "zorder": 2}
        options.update(kwargs)
        self._plot(**options)


@dataclass
class ContactLocationVertex(ContactLocation):
    index: int

    def _plot(self, **kwargs):
        self.body.geometry.plot_vertex(self.index, **kwargs)

    @property
    def vertex(self):
        return self.body.geometry.vertices[self.index]

    @property
    def adj_faces(self):
        if self.body.dim != 2:
            raise NotImplementedError
        adj_face_before = ContactLocationFace(
            self.body, (self.index - 1) % self.body.n_vertices
        )
        adj_face_after = ContactLocationFace(self.body, self.index)
        return (adj_face_before, adj_face_after)


@dataclass
class ContactLocationFace(ContactLocation):
    halfspace_index: int

    def _plot(self, **kwargs):
        self.body.geometry.plot_vertex(self.adj_vertex_indices[0], **kwargs)
        self.body.geometry.plot_vertex(self.adj_vertex_indices[1], **kwargs)
        self.body.geometry.plot_face(self.halfspace_index, **kwargs)

    @property
    def normal(self):
        return self.body.geometry.set.A()[self.halfspace_index]

    @property
    def b(self):
        return self.body.geometry.set.b()[self.halfspace_index]

    @property
    def unit_normal(self):
        return self.normal / np.linalg.norm(self.normal)

    @property
    def adj_vertex_indices(self):
        if self.body.dim != 2:
            raise NotImplementedError
        return np.array(
            [
                self.halfspace_index,
                (self.halfspace_index + 1) % self.body.geometry.vertices.shape[0],
            ]
        )

    @property
    def vec_center_to_face(self):
        """Vector from the center of the body to the face in the direction of negative normal of the face
        (normal of the face points outwards, away from the center of the body)"""
        vec_vertex_center = (
            self.body.geometry.center
            - self.body.geometry.vertices[self.adj_vertex_indices[0]]
        )
        dist = np.dot(vec_vertex_center, -self.unit_normal)
        return dist * self.unit_normal


# Utility functions
def is_possible_face_face_contact(
    face_a: ContactLocationFace, face_b: ContactLocationFace
):
    """Check if two faces can be in contact"""
    return np.isclose(np.dot(face_a.unit_normal, face_b.unit_normal), -1)


def is_possible_face_vertex_contact(
    face: ContactLocationFace, vertex: ContactLocationVertex
):
    """Check if a face and a vertex can be in contact"""
    adj_face_before, adj_face_after = vertex.adj_faces

    before_angle = counter_clockwise_angle_between(
        face.unit_normal, adj_face_before.unit_normal
    )
    after_angle = counter_clockwise_angle_between(
        face.unit_normal, adj_face_after.unit_normal
    )

    # If the faces are parallel or anti-parallel, then the vertex is not in contact
    if (
        np.isclose(before_angle, np.pi)
        or np.isclose(after_angle, np.pi)
        or np.isclose(before_angle, -np.pi)
        or np.isclose(after_angle, -np.pi)
        or np.isclose(before_angle, 0)
        or np.isclose(after_angle, 0)
    ):
        res = False
    # If the above is not true,
    # AND the angle between the face normal and adjacent face to the "right"
    # (where the adjacent normals are pointing up, and the vertex is in between them)
    # AKA "before" (in a counter clockwise sense) 's normal has a negative angle
    # AND the angle between the face normal and adjacent face to the "left"
    # AKA "after" (in a counter clockwise sense) 's normal has a positive angle
    elif after_angle < 0 and before_angle > 0:
        res = True
    else:
        res = False
    return res
