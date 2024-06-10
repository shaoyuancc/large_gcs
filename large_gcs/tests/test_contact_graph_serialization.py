import numpy as np

from large_gcs.contact.contact_location import ContactLocationFace
from large_gcs.contact.contact_pair_mode import NoContactPairMode
from large_gcs.contact.rigid_body import MobilityType, RigidBody
from large_gcs.geometry.polyhedron import Polyhedron
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph_generators.contact_graph_generator import ContactGraphGeneratorParams

eps = 1e-6


def test_serialize_deserialize_contact_pair_mode():
    body_a_vert = np.array([[1, 1], [1, 0], [0, 0], [0, 1]])
    body_b_vert = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) + np.array([1.5, 0.5])

    body_a = RigidBody(
        "obj_a",
        Polyhedron.from_vertices(body_a_vert),
        MobilityType.STATIC,
    )
    body_b = RigidBody(
        "obj_b",
        Polyhedron.from_vertices(body_b_vert),
        MobilityType.UNACTUATED,
    )
    body_a.plot()
    body_b.plot()
    contact_loc_a = ContactLocationFace(body_a, 3)
    contact_loc_b = ContactLocationFace(body_b, 1)

    contact_pair_mode = NoContactPairMode(body_a, body_b, contact_loc_a, contact_loc_b)
    params = contact_pair_mode.params
    contact_pair_mode_deserialized = params.type(
        body_a,
        body_b,
        params.contact_location_a_type(body_a, params.contact_location_a_index),
        params.contact_location_b_type(body_b, params.contact_location_b_index),
    )
    assert contact_pair_mode_deserialized.params == params

def test_load_cg_simple_1_1():
    for graph_name in ["cg_simple_1_1"]:
        graph_file = ContactGraphGeneratorParams.graph_file_path_from_name(
            graph_name
        )
        cg = ContactGraph.load_from_file(
            graph_file,
            should_use_l1_norm_vertex_cost=True,
        )
