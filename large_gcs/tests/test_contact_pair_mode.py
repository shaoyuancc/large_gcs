import numpy as np
import pytest

from large_gcs.contact.contact_location import (
    ContactLocationFace,
    ContactLocationVertex,
)
from large_gcs.contact.contact_pair_mode import (
    create_movable_face_face_signed_dist_surrog_exprs,
    create_movable_face_vert_signed_dist_surrog_exprs,
    create_static_face_movable_face_horizontal_bounds_formulas,
    create_static_face_movable_face_signed_dist_surrog_exprs,
    create_static_face_movable_vert_signed_dist_surrog_exprs,
    create_static_vert_movable_face_horizontal_bounds_formulas,
    create_static_vert_movable_face_signed_dist_surrog_exprs,
)
from large_gcs.contact.rigid_body import MobilityType, RigidBody
from large_gcs.geometry.polyhedron import Polyhedron

eps = 1e-6


def test_create_static_face_movable_face_signed_dist_surrog_square():
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

    # contact_loc_a.plot()
    # contact_loc_b.plot()
    # plt.show()
    exprs = create_static_face_movable_face_signed_dist_surrog_exprs(
        contact_loc_a, contact_loc_b
    )

    vals = [1.5, 0]
    for expr, vars in zip(exprs, body_b.vars_pos.T):
        assert expr.is_polynomial()
        env = dict(zip(vars, vals))
        dist_surrog = expr.Evaluate(env)
        assert np.isclose(dist_surrog, 0)

    vals = [15, 11]
    for expr, vars in zip(exprs, body_b.vars_pos.T):
        env = dict(zip(vars, vals))
        dist_surrog = expr.Evaluate(env)
        assert dist_surrog > 0

    vals = [-1, 11]
    for expr, vars in zip(exprs, body_b.vars_pos.T):
        env = dict(zip(vars, vals))
        dist_surrog = expr.Evaluate(env)
        assert dist_surrog < 0


def test_create_static_face_movable_face_signed_dist_surrog_triangle():
    body_a = RigidBody(
        "obj_a",
        Polyhedron(A=[[1.5, 1], [-1, 0], [0, -1]], b=[4, 0, 0]),
        MobilityType.STATIC,
    )
    body_b = RigidBody(
        "obj_b",
        Polyhedron(A=[[-1.5, -1], [1, 0], [0, 1]], b=[-4, 3, 5]),
        MobilityType.UNACTUATED,
    )
    contact_loc_a = ContactLocationFace(body_a, 0)
    contact_loc_b = ContactLocationFace(body_b, 0)
    # contact_loc_a.plot()
    # contact_loc_b.plot()
    # plt.show()
    exprs = create_static_face_movable_face_signed_dist_surrog_exprs(
        contact_loc_a, contact_loc_b
    )
    vals = body_b.geometry.center
    for expr, vars in zip(exprs, body_b.vars_pos.T):
        assert expr.is_polynomial()
        env = dict(zip(vars, vals))
        dist_surrog = expr.Evaluate(env)
        assert np.isclose(dist_surrog, 0)

    vals = body_b.geometry.center + np.array([1, 1])
    for expr, vars in zip(exprs, body_b.vars_pos.T):
        env = dict(zip(vars, vals))
        dist_surrog = expr.Evaluate(env)
        assert dist_surrog > 0

    vals = body_b.geometry.center + np.array([-1, -1])
    for expr, vars in zip(exprs, body_b.vars_pos.T):
        env = dict(zip(vars, vals))
        dist_surrog = expr.Evaluate(env)
        assert dist_surrog < 0


def test_create_static_face_movable_face_signed_dist_surrog_fails_not_opposing():
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
    contact_loc_b = ContactLocationFace(body_b, 3)
    for i in range(3):
        contact_loc_a = ContactLocationFace(body_a, i)
        with pytest.raises(AssertionError):
            create_static_face_movable_face_signed_dist_surrog_exprs(
                contact_loc_a, contact_loc_b
            )


def test_create_static_face_movable_vertex_signed_dist_surrog_square():
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
    contact_face_b = ContactLocationFace(body_b, 3)
    contact_locs_b = [
        ContactLocationVertex(body_b, i) for i in contact_face_b.adj_vertex_indices
    ]
    for contact_loc_b in contact_locs_b:
        exprs = create_static_face_movable_vert_signed_dist_surrog_exprs(
            contact_loc_a, contact_loc_b
        )
        vals = [1.5, 0]
        for expr, vars in zip(exprs, body_b.vars_pos.T):
            assert expr.is_polynomial()
            env = dict(zip(vars, vals))
            dist_surrog = expr.Evaluate(env)
            assert np.isclose(dist_surrog, 0)

        vals = [15, 11]
        for expr, vars in zip(exprs, body_b.vars_pos.T):
            env = dict(zip(vars, vals))
            dist_surrog = expr.Evaluate(env)
            assert dist_surrog > 0

        vals = [-1, 11]
        for expr, vars in zip(exprs, body_b.vars_pos.T):
            env = dict(zip(vars, vals))
            dist_surrog = expr.Evaluate(env)
            assert dist_surrog < 0


def test_create_static_vert_movable_face_signed_dist_surrog_square():
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
    contact_face_a = ContactLocationFace(body_a, 3)
    contact_loc_b = ContactLocationFace(body_b, 3)
    contact_locs_a = [
        ContactLocationVertex(body_a, i) for i in contact_face_a.adj_vertex_indices
    ]
    for contact_loc_a in contact_locs_a:
        exprs = create_static_vert_movable_face_signed_dist_surrog_exprs(
            contact_loc_a, contact_loc_b
        )
        vals = [1.5, 0]
        for expr, vars in zip(exprs, body_b.vars_pos.T):
            assert expr.is_polynomial()
            env = dict(zip(vars, vals))
            dist_surrog = expr.Evaluate(env)
            assert np.isclose(dist_surrog, 0)

        vals = [15, 11]
        for expr, vars in zip(exprs, body_b.vars_pos.T):
            env = dict(zip(vars, vals))
            dist_surrog = expr.Evaluate(env)
            assert dist_surrog > 0

        vals = [-1, 11]
        for expr, vars in zip(exprs, body_b.vars_pos.T):
            env = dict(zip(vars, vals))
            dist_surrog = expr.Evaluate(env)
            assert dist_surrog < 0


def test_create_movable_face_face_signed_dist_surrog_triangle():
    body_a = RigidBody(
        "obj_a",
        Polyhedron(A=[[1.5, 1], [-1, 0], [0, -1]], b=[4, 0, 0]),
        MobilityType.UNACTUATED,
    )
    body_b = RigidBody(
        "obj_b",
        Polyhedron(A=[[-1.5, -1], [1, 0], [0, 1]], b=[-4, 3, 5]),
        MobilityType.ACTUATED,
    )
    contact_loc_a = ContactLocationFace(body_a, 0)
    contact_loc_b = ContactLocationFace(body_b, 0)

    exprs = create_movable_face_face_signed_dist_surrog_exprs(
        contact_loc_a, contact_loc_b
    )
    vals = np.hstack((body_a.geometry.center, body_b.geometry.center))
    exprs_vars = zip(exprs, np.hstack((body_a.vars_pos.T, body_b.vars_pos.T)))

    for expr, vars in exprs_vars:
        assert expr.is_polynomial()
        env = dict(zip(vars, vals))
        dist_surrog = expr.Evaluate(env)
        assert np.isclose(dist_surrog, 0)

    vals = np.hstack(
        (body_a.geometry.center, body_b.geometry.center + np.array([1, 1]))
    )
    for expr, vars in exprs_vars:
        env = dict(zip(vars, vals))
        dist_surrog = expr.Evaluate(env)
        assert dist_surrog > 0

    vals = np.hstack(
        (body_a.geometry.center, body_b.geometry.center + np.array([-1, -1]))
    )
    for expr, vars in exprs_vars:
        env = dict(zip(vars, vals))
        dist_surrog = expr.Evaluate(env)
        assert dist_surrog < 0


def test_create_movable_face_vert_signed_dist_surrog_triangle():
    body_a = RigidBody(
        "obj_a",
        Polyhedron(A=[[1.5, 1], [-1, 0], [0, -1]], b=[4, 0, 0]),
        MobilityType.UNACTUATED,
    )
    body_b = RigidBody(
        "obj_b",
        Polyhedron(A=[[-1.5, -1], [1, 0], [0, 1]], b=[-4, 3, 5]),
        MobilityType.ACTUATED,
    )
    contact_loc_a = ContactLocationFace(body_a, 0)
    contact_loc_b = ContactLocationVertex(body_b, 0)

    exprs = create_movable_face_vert_signed_dist_surrog_exprs(
        contact_loc_a, contact_loc_b
    )
    vals = np.hstack((body_a.geometry.center, body_b.geometry.center))
    exprs_vars = zip(exprs, np.hstack((body_a.vars_pos.T, body_b.vars_pos.T)))

    for expr, vars in exprs_vars:
        assert expr.is_polynomial()
        env = dict(zip(vars, vals))
        dist_surrog = expr.Evaluate(env)
        assert np.isclose(dist_surrog, 0)

    vals = np.hstack(
        (body_a.geometry.center, body_b.geometry.center + np.array([1, 1]))
    )
    for expr, vars in exprs_vars:
        env = dict(zip(vars, vals))
        dist_surrog = expr.Evaluate(env)
        assert dist_surrog > 0

    vals = np.hstack(
        (body_a.geometry.center, body_b.geometry.center + np.array([-1, -1]))
    )
    for expr, vars in exprs_vars:
        env = dict(zip(vars, vals))
        dist_surrog = expr.Evaluate(env)
        assert dist_surrog < 0


def test_create_movable_vert_face_signed_dist_surrog_triangle():
    body_a = RigidBody(
        "obj_a",
        Polyhedron(A=[[1.5, 1], [-1, 0], [0, -1]], b=[4, 0, 0]),
        MobilityType.UNACTUATED,
    )
    body_b = RigidBody(
        "obj_b",
        Polyhedron(A=[[-1.5, -1], [1, 0], [0, 1]], b=[-4, 3, 5]),
        MobilityType.ACTUATED,
    )
    contact_loc_a = ContactLocationFace(body_a, 0)
    contact_loc_b = ContactLocationVertex(body_b, 0)

    exprs = create_movable_face_vert_signed_dist_surrog_exprs(
        contact_loc_a, contact_loc_b
    )
    vals = np.hstack((body_a.geometry.center, body_b.geometry.center))
    exprs_vars = zip(exprs, np.hstack((body_a.vars_pos.T, body_b.vars_pos.T)))

    for expr, vars in exprs_vars:
        assert expr.is_polynomial()
        env = dict(zip(vars, vals))
        dist_surrog = expr.Evaluate(env)
        assert np.isclose(dist_surrog, 0)

    vals = np.hstack(
        (body_a.geometry.center, body_b.geometry.center + np.array([1, 1]))
    )
    for expr, vars in exprs_vars:
        env = dict(zip(vars, vals))
        dist_surrog = expr.Evaluate(env)
        assert dist_surrog > 0

    vals = np.hstack(
        (body_a.geometry.center, body_b.geometry.center + np.array([-1, -1]))
    )
    for expr, vars in exprs_vars:
        env = dict(zip(vars, vals))
        dist_surrog = expr.Evaluate(env)
        assert dist_surrog < 0


def test_create_static_face_movable_face_horizontal_bounds_square():
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

    formulas = create_static_face_movable_face_horizontal_bounds_formulas(
        contact_loc_a, contact_loc_b
    )
    y_vals = np.linspace(-0.3 + eps, 1.3 - eps, 20)
    x_vals = [1.5] * 20
    vals_list = list(zip(x_vals, y_vals))
    for vals in vals_list:
        vars = body_b.vars_pos.T.flatten()
        env = dict(zip(vars, vals * body_b.n_pos_points))
        for formula in formulas:
            assert formula.Evaluate(env)


def test_create_static_vertex_movable_face_horizontal_bounds_triangle():
    body_a = RigidBody(
        "obj_a",
        Polyhedron.from_vertices([[0, 0], [1, 0], [0, -1]]),
        MobilityType.STATIC,
    )
    body_b = RigidBody(
        "obj_b",
        Polyhedron.from_vertices([[-1, -1], [-1.5, -0.5], [-1.2, -1.5]]),
        MobilityType.UNACTUATED,
    )
    # First
    contact_loc_a = ContactLocationVertex(body_a, 0)
    contact_loc_b = ContactLocationFace(body_b, 0)

    formulas = create_static_vert_movable_face_horizontal_bounds_formulas(
        contact_loc_a, contact_loc_b
    )
    vals = [-0.63333045, -0.59998453]
    vars = body_b.vars_pos.T.flatten()
    env = dict(zip(vars, vals * body_b.n_pos_points))
    res = [formula.Evaluate(env) for formula in formulas]
    assert not all(res)

    # Second
    contact_loc_a = ContactLocationVertex(body_a, 2)
    contact_loc_b = ContactLocationFace(body_b, 2)

    formulas = create_static_vert_movable_face_horizontal_bounds_formulas(
        contact_loc_a, contact_loc_b
    )
    vals = [0.12666955, 0.90001547]
    vars = body_b.vars_pos.T.flatten()
    env = dict(zip(vars, vals * body_b.n_pos_points))
    res = [formula.Evaluate(env) for formula in formulas]
    assert not all(res)
