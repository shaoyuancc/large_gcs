import numpy as np
import matplotlib.pyplot as plt
from large_gcs.contact.contact_location import (
    ContactLocationFace,
    ContactLocationVertex,
)
from large_gcs.contact.contact_set import *
from large_gcs.contact.contact_pair_mode import *
from large_gcs.contact.rigid_body import MobilityType, RigidBody
from large_gcs.geometry.geometry_utils import plot_vector
from large_gcs.geometry.polyhedron import Polyhedron
import pytest


def test_create_incontactpairmode_static_tri_movable_tri():
    n_pos_points = 10
    body_a = RigidBody(
        "obj_a",
        Polyhedron.from_vertices([[0, 0], [1, 0], [0, -1]]),
        MobilityType.STATIC,
        n_pos_points=n_pos_points,
    )
    body_b = RigidBody(
        "obj_b",
        Polyhedron.from_vertices([[-1, -1], [-1.5, -0.5], [-1.2, -1.5]]),
        MobilityType.UNACTUATED,
        n_pos_points=n_pos_points,
    )
    contact_loc_a = ContactLocationVertex(body_a, 0)
    contact_loc_b = ContactLocationFace(body_b, 0)
    contact_pair_mode = InContactPairMode(body_a, body_b, contact_loc_a, contact_loc_b)

    all_vars = ContactSet.flatten_set_vars(body_b.vars_pos)
    contact_set = ContactSet([contact_pair_mode], all_vars)
    xy = [0.12666955, 0.90001547]
    vals = np.repeat(xy, n_pos_points)
    assert not contact_set.set.PointInSet(vals)
