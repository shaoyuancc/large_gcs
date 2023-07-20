import numpy as np
import matplotlib.pyplot as plt
from large_gcs.contact.contact_location import (
    ContactLocationFace,
    ContactLocationVertex,
)
from large_gcs.contact.contact_pair_mode import *
from large_gcs.contact.rigid_body import MobilityType, RigidBody
from large_gcs.geometry.polyhedron import Polyhedron
import pytest

eps = 1e-6


def test_serialize_deserialize_contact_pair_mode():
    body_a = RigidBody(
        "obj_a",
        Polyhedron.from_vertices([[0, 0], [1, 0], [1, 1], [0, 1]]),
        MobilityType.STATIC,
    )
    body_b_vert = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) + np.array([1.5, 0.5])
    body_b = RigidBody(
        "obj_b",
        Polyhedron.from_vertices(body_b_vert),
        MobilityType.UNACTUATED,
    )
    contact_loc_a = ContactLocationFace(body_a, 3)
    contact_loc_b = ContactLocationFace(body_b, 3)

    contact_pair_mode = NoContactPairMode(body_a, body_b, contact_loc_a, contact_loc_b)
    params = contact_pair_mode.params
    print(params)
    contact_pair_mode_deserialized = params.type(
        body_a,
        body_b,
        params.contact_location_a_type(body_a, params.contact_location_a_index),
        params.contact_location_b_type(body_b, params.contact_location_b_index),
    )
    assert contact_pair_mode_deserialized.params == params
