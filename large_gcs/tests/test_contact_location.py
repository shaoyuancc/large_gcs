import numpy as np
import matplotlib.pyplot as plt

from large_gcs.contact.contact_location import (
    ContactLocationFace,
    ContactLocationVertex,
)
from large_gcs.contact.rigid_body import MobilityType, RigidBody
from large_gcs.geometry.geometry_utils import plot_vector
from large_gcs.geometry.polyhedron import Polyhedron


def test_vec_center_to_face_square():
    body = RigidBody(
        "obj",
        Polyhedron.from_vertices([[0, 0], [1, 0], [1, 1], [0, 1]]),
        MobilityType.UNACTUATED,
    )
    contact_loc_face = ContactLocationFace(body, 0)
    # contact_loc_face.plot()
    # plt.show()
    assert np.allclose(contact_loc_face.vec_center_to_face, np.array([0, 0.5]))


def test_vec_center_to_face_triangle_dist():
    body = RigidBody(
        "obj",
        Polyhedron.from_vertices([[0, 0], [1, -0.5], [3, 1]]),
        MobilityType.UNACTUATED,
    )
    contact_loc_face = ContactLocationFace(body, 0)
    # print(f"center: {body.geometry.center}, normal: {contact_loc_face.normal}, b: {contact_loc_face.b}")
    # body.geometry.plot(mark_center=True)
    # contact_loc_face.plot()
    # plot_vector(contact_loc_face.vec_center_to_face, body.geometry.center, color="g")
    # plt.show()
    assert np.isclose(
        np.linalg.norm(contact_loc_face.vec_center_to_face), 0.2635, atol=1e-4
    )
