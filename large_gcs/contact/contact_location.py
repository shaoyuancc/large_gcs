from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from large_gcs.contact.rigid_body import RigidBody
from large_gcs.geometry.geometry_utils import counter_clockwise_angle_between


@dataclass
class ContactLocation(ABC):
    body: RigidBody
    index: int

    def plot(self, **kwargs):
        if self.body.dim != 2:
            raise NotImplementedError
        self.body.geometry.plot()
        options = {"color": "r", "zorder": 2}
        options.update(kwargs)
        self._plot(**options)

    @property
    @abstractmethod
    def compact_name(self):
        pass


@dataclass
class ContactLocationVertex(ContactLocation):
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

    @property
    def p_CV(self):
        """Position vector from the center of the body to the vertex."""
        return self.vertex - self.body.geometry.center

    @property
    def compact_name(self):
        return f"v{self.index}"


@dataclass
class ContactLocationFace(ContactLocation):
    def _plot(self, **kwargs):
        self.body.geometry.plot_vertex(self.adj_vertex_indices[0], **kwargs)
        self.body.geometry.plot_vertex(self.adj_vertex_indices[1], **kwargs)
        self.body.geometry.plot_face(self.index, **kwargs)

    @property
    def normal(self):
        return self.body.geometry.set.A()[self.index]

    @property
    def b(self):
        return self.body.geometry.set.b()[self.index]

    @property
    def unit_normal(self):
        return self.normal / np.linalg.norm(self.normal)

    @property
    def adj_vertex_indices(self):
        """Vertices are ordered counter-clockwise, so index 0 is the vertex
        before the face AKA 'right', and index 1 is the vertex after the face
        AKA 'left'."""
        if self.body.dim != 2:
            raise NotImplementedError
        return np.array(
            [
                self.index,
                (self.index + 1) % self.body.geometry.vertices.shape[0],
            ]
        )

    @property
    def adj_vertices(self):
        return self.body.geometry.vertices[self.adj_vertex_indices]

    @property
    def p_CF(self):
        """Vector from the center of the body to the face in the direction of
        negative normal of the face (normal of the face points outwards, away
        from the center of the body)"""
        vec_vertex_center = (
            self.body.geometry.center
            - self.body.geometry.vertices[self.adj_vertex_indices[0]]
        )
        dist = np.dot(vec_vertex_center, -self.unit_normal)
        return dist * self.unit_normal

    @property
    def p_CVright(self):
        """Position vector from the center of the body to the vertex before the
        face."""
        return self.adj_vertices[0] - self.body.geometry.center

    @property
    def p_CVleft(self):
        """Position vector from the center of the body to the vertex after the
        face."""
        return self.adj_vertices[1] - self.body.geometry.center

    @property
    def length(self):
        """Length of the face."""
        return np.linalg.norm(self.adj_vertices[1] - self.adj_vertices[0])

    @property
    def compact_name(self):
        return f"f{self.index}"


# Utility functions
def is_possible_face_face_contact(
    face_a: ContactLocationFace, face_b: ContactLocationFace
):
    """Check if two faces can be in contact."""
    return np.isclose(np.dot(face_a.unit_normal, face_b.unit_normal), -1)


def is_possible_face_vertex_contact(
    face: ContactLocationFace, vertex: ContactLocationVertex
):
    """Check if a face and a vertex can be in contact."""
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
