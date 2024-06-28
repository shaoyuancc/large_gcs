import logging

import pytest
from pydrake.all import HPolyhedron

from large_gcs.algorithms.gcs_star import GcsStar
from large_gcs.algorithms.search_algorithm import AlgMetrics, SearchNode
from large_gcs.cost_estimators.shortcut_edge_ce import ShortcutEdgeCE
from large_gcs.domination_checkers.ah_containment_last_pos import (
    ReachesCheaperLastPosContainment,
    ReachesNewLastPosContainment,
)
from large_gcs.domination_checkers.reaches_cheaper_containment import (
    ReachesCheaperContainment,
)
from large_gcs.domination_checkers.reaches_new_containment import ReachesNewContainment
from large_gcs.graph.contact_cost_constraint_factory import (
    contact_shortcut_edge_l1norm_cost_factory_obj_weighted,
)
from large_gcs.graph.incremental_contact_graph import IncrementalContactGraph
from large_gcs.graph_generators.contact_graph_generator import (
    ContactGraphGeneratorParams,
)
from large_gcs.graph_generators.hor_vert_gcs import create_polyhedral_hor_vert_b_graph
from large_gcs.graph_generators.one_dimensional_gcs import create_simple_1d_graph
from large_gcs.utils.utils import all_lists_equal

logger = logging.getLogger(__name__)


def test_reaches_new_containment_polyhedral_hor_vert_b_graph():
    G = create_polyhedral_hor_vert_b_graph()
    domination_checker = ReachesNewContainment(graph=G, containment_condition=-1)
    domination_checker.set_alg_metrics(AlgMetrics())
    n_x = SearchNode.from_vertex_path(["s", "p1", "p2"])
    n_y = SearchNode.from_vertex_path(["s", "p6", "p7", "p2"])
    n_z = SearchNode.from_vertex_path(["s", "p8", "p9", "p2"])

    assert domination_checker.is_dominated(candidate_node=n_x, alternate_nodes=[n_y])
    assert not domination_checker.is_dominated(
        candidate_node=n_y, alternate_nodes=[n_x]
    )
    assert domination_checker.is_dominated(candidate_node=n_z, alternate_nodes=[n_x])

    # The union of n_b, n_c, and n_d does dominate n_a,
    # but individually they do not, so this check should return false.
    n_a = SearchNode.from_vertex_path(["s", "p0", "p2"])
    n_b = SearchNode.from_vertex_path(["s", "p3", "p2"])
    n_c = SearchNode.from_vertex_path(["s", "p4", "p2"])
    n_d = SearchNode.from_vertex_path(["s", "p5", "p2"])
    assert not domination_checker.is_dominated(
        candidate_node=n_a, alternate_nodes=[n_b, n_c, n_d]
    )


def test_reaches_cheaper_containment_polyhedral_hor_vert_b_graph():
    G = create_polyhedral_hor_vert_b_graph()
    domination_checker = ReachesCheaperContainment(graph=G, containment_condition=-1)
    domination_checker.set_alg_metrics(AlgMetrics())
    n_x = SearchNode.from_vertex_path(["s", "p1", "p2"])
    n_y = SearchNode.from_vertex_path(["s", "p6", "p7", "p2"])
    n_z = SearchNode.from_vertex_path(["s", "p8", "p9", "p2"])

    assert not domination_checker.is_dominated(
        candidate_node=n_x, alternate_nodes=[n_y]
    )
    assert not domination_checker.is_dominated(
        candidate_node=n_y, alternate_nodes=[n_x]
    )
    assert domination_checker.is_dominated(candidate_node=n_z, alternate_nodes=[n_x])

    # The union of n_b, n_c, and n_d does dominate n_a,
    # but individually they do not, so this check should return false.
    n_a = SearchNode.from_vertex_path(["s", "p0", "p2"])
    n_b = SearchNode.from_vertex_path(["s", "p3", "p2"])
    n_c = SearchNode.from_vertex_path(["s", "p4", "p2"])
    n_d = SearchNode.from_vertex_path(["s", "p5", "p2"])
    assert not domination_checker.is_dominated(
        candidate_node=n_a, alternate_nodes=[n_b, n_c, n_d]
    )


def test_reaches_cheaper_containment_simple_1d_graph():
    G = create_simple_1d_graph()
    domination_checker = ReachesCheaperContainment(graph=G, containment_condition=-1)
    domination_checker.set_alg_metrics(AlgMetrics())
    n = SearchNode.from_vertex_path(["s", "t"])
    n_prime = SearchNode.from_vertex_path(["p0", "t"])

    assert domination_checker.is_dominated(candidate_node=n, alternate_nodes=[n_prime])
    assert not domination_checker.is_dominated(
        candidate_node=n_prime, alternate_nodes=[n]
    )


def test_nullspace_polyhedron_and_transformation_from_HPoly_and_T_correct_shapes_cg_trichal4():
    graph_file = ContactGraphGeneratorParams.inc_graph_file_path_from_name(
        "cg_trichal4"
    )
    cg = IncrementalContactGraph.load_from_file(
        graph_file,
        should_incl_simul_mode_switches=False,
        should_add_const_edge_cost=True,
        should_add_gcs=True,
        should_use_l1_norm_vertex_cost=True,
    )
    domination_checker = ReachesNewContainment(
        graph=cg, containment_condition=-1, construct_path_from_nullspaces=True
    )
    domination_checker.set_alg_metrics(AlgMetrics())

    # fmt: off
    cand_path = ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f0-rob0_v2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_v0-rob0_f1', 'IC|obj0_f0-rob0_v2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f2-rob0_v1', 'IC|obj0_f0-rob0_v2')"]
    alt_path = ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f0-rob0_v2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f2-rob0_v1', 'IC|obj0_f0-rob0_v2')"]
    # fmt: on

    # Load these two paths into the graph
    cg.add_vertex_path_to_graph(cand_path)
    cg.add_vertex_path_to_graph(alt_path)

    node = SearchNode.from_vertex_path(cand_path)
    alt_node = SearchNode.from_vertex_path(alt_path)
    prog, full_dim = domination_checker.get_nullspace_path_mathematical_program(node)
    h_poly = HPolyhedron(prog)
    T_H, t_H = domination_checker.get_nullspace_H_transformation(
        node, full_dim=full_dim
    )
    (
        K,
        k,
        T,
        t,
        _,
    ) = domination_checker._nullspace_polyhedron_and_transformation_from_HPoly_and_T(
        h_poly, T_H, t_H
    )
    logger.debug(
        f"K: {K.shape}, k: {k.shape}, T: {T.shape}, t: {t.shape}, T_H: {T_H.shape}, t_H: {t_H.shape}"
    )
    assert t.shape == t_H.shape
    domination_checker.is_dominated(candidate_node=node, alternate_nodes=[alt_node])


def test_last_pos_nullspace_polyhedron_and_transformation_from_HPoly_and_T_correct_shapes_cg_trichal4():
    graph_file = ContactGraphGeneratorParams.inc_graph_file_path_from_name(
        "cg_trichal4"
    )
    cg = IncrementalContactGraph.load_from_file(
        graph_file,
        should_incl_simul_mode_switches=False,
        should_add_const_edge_cost=True,
        should_add_gcs=True,
        should_use_l1_norm_vertex_cost=True,
    )
    domination_checker = ReachesNewLastPosContainment(
        graph=cg, containment_condition=-1, construct_path_from_nullspaces=True
    )
    domination_checker.set_alg_metrics(AlgMetrics())

    # fmt: off
    cand_path = ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f0-rob0_v2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_v0-rob0_f1', 'IC|obj0_f0-rob0_v2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f2-rob0_v1', 'IC|obj0_f0-rob0_v2')"]
    alt_path = ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f0-rob0_v2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f2-rob0_v1', 'IC|obj0_f0-rob0_v2')"]
    # fmt: on

    # Load these two paths into the graph
    cg.add_vertex_path_to_graph(cand_path)
    cg.add_vertex_path_to_graph(alt_path)

    node = SearchNode.from_vertex_path(cand_path)
    alt_node = SearchNode.from_vertex_path(alt_path)
    prog, full_dim = domination_checker.get_nullspace_path_mathematical_program(node)
    h_poly = HPolyhedron(prog)
    T_H, t_H = domination_checker.get_nullspace_H_transformation(
        node, full_dim=full_dim
    )
    (
        K,
        k,
        T,
        t,
        _,
    ) = domination_checker._nullspace_polyhedron_and_transformation_from_HPoly_and_T(
        h_poly, T_H, t_H
    )
    logger.debug(
        f"K: {K.shape}, k: {k.shape}, T: {T.shape}, t: {t.shape}, T_H: {T_H.shape}, t_H: {t_H.shape}"
    )
    assert t.shape == t_H.shape
    domination_checker.is_dominated(candidate_node=node, alternate_nodes=[alt_node])


@pytest.mark.slow_test
def test_construct_path_from_nullspaces_reaches_new_cg_simple_4():
    graph_file = ContactGraphGeneratorParams.inc_graph_file_path_from_name(
        "cg_simple_4"
    )
    cg = IncrementalContactGraph.load_from_file(
        graph_file,
        should_incl_simul_mode_switches=False,
        should_add_const_edge_cost=True,
        should_add_gcs=True,
        should_use_l1_norm_vertex_cost=True,
    )
    expansion_orders = []
    for construct_path_from_nullspaces in [False, True]:
        domination_checker = ReachesNewContainment(
            graph=cg,
            containment_condition=-1,
            construct_path_from_nullspaces=construct_path_from_nullspaces,
        )
        domination_checker.set_alg_metrics(AlgMetrics())
        cost_estimator = ShortcutEdgeCE(
            graph=cg,
            shortcut_edge_cost_factory=contact_shortcut_edge_l1norm_cost_factory_obj_weighted,
            add_const_cost=True,
        )
        alg = GcsStar(
            graph=cg,
            cost_estimator=cost_estimator,
            domination_checker=domination_checker,
            save_expansion_order=True,
            heuristic_inflation_factor=10,
        )
        alg.run()
        expansion_orders.append(alg.alg_metrics.expansion_order)

    # Check that the expansion orders are the same
    assert all_lists_equal(expansion_orders)


@pytest.mark.slow_test
def test_construct_path_from_nullspaces_reaches_cheaper_cg_simple_4():
    graph_file = ContactGraphGeneratorParams.inc_graph_file_path_from_name(
        "cg_simple_4"
    )
    cg = IncrementalContactGraph.load_from_file(
        graph_file,
        should_incl_simul_mode_switches=False,
        should_add_const_edge_cost=True,
        should_add_gcs=True,
        should_use_l1_norm_vertex_cost=True,
    )
    expansion_orders = []
    for construct_path_from_nullspaces in [True, False]:
        domination_checker = ReachesCheaperContainment(
            graph=cg,
            containment_condition=-1,
            construct_path_from_nullspaces=construct_path_from_nullspaces,
        )
        domination_checker.set_alg_metrics(AlgMetrics())
        cost_estimator = ShortcutEdgeCE(
            graph=cg,
            shortcut_edge_cost_factory=contact_shortcut_edge_l1norm_cost_factory_obj_weighted,
            add_const_cost=True,
        )
        alg = GcsStar(
            graph=cg,
            cost_estimator=cost_estimator,
            domination_checker=domination_checker,
            save_expansion_order=True,
            heuristic_inflation_factor=10,
        )
        alg.run()
        expansion_orders.append(alg.alg_metrics.expansion_order)

    # Check that the expansion orders are the same
    assert all_lists_equal(expansion_orders)


@pytest.mark.slow_test
def test_construct_path_from_nullspaces_reaches_new_last_pos_cg_simple_4():
    graph_file = ContactGraphGeneratorParams.inc_graph_file_path_from_name(
        "cg_simple_4"
    )
    cg = IncrementalContactGraph.load_from_file(
        graph_file,
        should_incl_simul_mode_switches=False,
        should_add_const_edge_cost=True,
        should_add_gcs=True,
        should_use_l1_norm_vertex_cost=True,
    )
    expansion_orders = []
    for construct_path_from_nullspaces in [False, True]:
        domination_checker = ReachesNewLastPosContainment(
            graph=cg,
            containment_condition=-1,
            construct_path_from_nullspaces=construct_path_from_nullspaces,
        )
        domination_checker.set_alg_metrics(AlgMetrics())
        cost_estimator = ShortcutEdgeCE(
            graph=cg,
            shortcut_edge_cost_factory=contact_shortcut_edge_l1norm_cost_factory_obj_weighted,
            add_const_cost=True,
        )
        alg = GcsStar(
            graph=cg,
            cost_estimator=cost_estimator,
            domination_checker=domination_checker,
            save_expansion_order=True,
            heuristic_inflation_factor=10,
        )
        alg.run()
        expansion_orders.append(alg.alg_metrics.expansion_order)

    # Check that the expansion orders are the same
    assert all_lists_equal(expansion_orders)


@pytest.mark.slow_test
def test_construct_path_from_nullspaces_reaches_cheaper_last_pos_cg_simple_4():
    graph_file = ContactGraphGeneratorParams.inc_graph_file_path_from_name(
        "cg_simple_4"
    )
    cg = IncrementalContactGraph.load_from_file(
        graph_file,
        should_incl_simul_mode_switches=False,
        should_add_const_edge_cost=True,
        should_add_gcs=True,
        should_use_l1_norm_vertex_cost=True,
    )
    expansion_orders = []
    for construct_path_from_nullspaces in [True, False]:
        domination_checker = ReachesCheaperLastPosContainment(
            graph=cg,
            containment_condition=-1,
            construct_path_from_nullspaces=construct_path_from_nullspaces,
        )
        domination_checker.set_alg_metrics(AlgMetrics())
        cost_estimator = ShortcutEdgeCE(
            graph=cg,
            shortcut_edge_cost_factory=contact_shortcut_edge_l1norm_cost_factory_obj_weighted,
            add_const_cost=True,
        )
        alg = GcsStar(
            graph=cg,
            cost_estimator=cost_estimator,
            domination_checker=domination_checker,
            save_expansion_order=True,
            heuristic_inflation_factor=10,
        )
        alg.run()
        expansion_orders.append(alg.alg_metrics.expansion_order)

    # Check that the expansion orders are the same
    assert all_lists_equal(expansion_orders)


def test_reaches_new_polyhedrons_are_the_same_cg_trichal4():
    """Compare the path polyhedrons created with nullspaces, and with full
    sets."""

    graph_file = ContactGraphGeneratorParams.inc_graph_file_path_from_name(
        "cg_trichal4"
    )
    cg = IncrementalContactGraph.load_from_file(
        graph_file,
        should_incl_simul_mode_switches=False,
        should_add_const_edge_cost=True,
        should_add_gcs=True,
        should_use_l1_norm_vertex_cost=True,
    )
    # fmt: off
    expansion_order = [['source'], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_v1-obj0_f3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_v0-rob0_f1')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f1-rob0_f2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f0-rob0_v2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f0-rob0_v2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_v0-rob0_f1', 'IC|obj0_f0-rob0_v2')"]]
    # fmt: on
    dom_ns = ReachesNewContainment(graph=cg, construct_path_from_nullspaces=True)
    dom_ns.set_alg_metrics(AlgMetrics())
    dom_fs = ReachesNewContainment(graph=cg, construct_path_from_nullspaces=False)
    dom_fs.set_alg_metrics(AlgMetrics())
    for path in expansion_order[1:-1]:
        cg.add_vertex_path_to_graph(path)
        node = SearchNode.from_vertex_path(path)
        AH_poly_ns, _ = dom_ns._create_path_AH_polytope_from_nullspace_sets(node)
        AH_poly_fs, _ = dom_fs._create_path_AH_polytope_from_full_sets(node)
        # They should be the same polyhedrons
        assert dom_fs.is_contained_in(AH_poly_ns, AH_poly_fs)
        assert dom_fs.is_contained_in(AH_poly_fs, AH_poly_ns)


def test_reaches_cheaper_polyhedrons_are_the_same_cg_trichal4():
    """Compare the path polyhedrons created with nullspaces, and with full
    sets."""

    graph_file = ContactGraphGeneratorParams.inc_graph_file_path_from_name(
        "cg_trichal4"
    )
    cg = IncrementalContactGraph.load_from_file(
        graph_file,
        should_incl_simul_mode_switches=False,
        should_add_const_edge_cost=True,
        should_add_gcs=True,
        should_use_l1_norm_vertex_cost=True,
    )
    # fmt: off
    expansion_order = [['source'], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_v1-obj0_f3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_v0-rob0_f1')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f1-rob0_f2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f0-rob0_v2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f0-rob0_v2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_v0-rob0_f1', 'IC|obj0_f0-rob0_v2')"]]
    # fmt: on
    dom_ns = ReachesCheaperContainment(graph=cg, construct_path_from_nullspaces=True)
    dom_ns.set_alg_metrics(AlgMetrics())
    dom_fs = ReachesCheaperContainment(graph=cg, construct_path_from_nullspaces=False)
    dom_fs.set_alg_metrics(AlgMetrics())
    for path in expansion_order[1:-1]:
        cg.add_vertex_path_to_graph(path)
        node = SearchNode.from_vertex_path(path)
        AH_poly_ns, _ = dom_ns._create_path_AH_polytope_from_nullspace_sets(node)
        AH_poly_fs, _ = dom_fs._create_path_AH_polytope_from_full_sets(node)
        # They should be the same polyhedrons
        assert dom_fs.is_contained_in(AH_poly_ns, AH_poly_fs)
        assert dom_fs.is_contained_in(AH_poly_fs, AH_poly_ns)


def test_last_pos_polyhedrons_are_the_same_cg_trichal4():
    """Compare the path polyhedrons created with nullspaces, and with full
    sets."""

    graph_file = ContactGraphGeneratorParams.inc_graph_file_path_from_name(
        "cg_trichal4"
    )
    cg = IncrementalContactGraph.load_from_file(
        graph_file,
        should_incl_simul_mode_switches=False,
        should_add_const_edge_cost=True,
        should_add_gcs=True,
        should_use_l1_norm_vertex_cost=True,
    )
    # fmt: off
    expansion_order = [['source'], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_v1-obj0_f3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_v0-rob0_f1')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f1-rob0_f2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f0-rob0_v2')"], ['source', "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f3-rob0_v1')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'NC|obj0_f2-rob0_f0')", "('NC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('IC|obs0_f0-obj0_v3', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_v0-obj0_f2', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f1-rob0_f2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_f0-rob0_v2', 'IC|obj0_f0-rob0_v2')", "('NC|obs0_f2-obj0_f1', 'NC|obs0_v0-rob0_f1', 'IC|obj0_f0-rob0_v2')"]]
    # fmt: on
    dom_ns = ReachesNewLastPosContainment(graph=cg, construct_path_from_nullspaces=True)
    dom_ns.set_alg_metrics(AlgMetrics())
    dom_fs = ReachesNewLastPosContainment(
        graph=cg, construct_path_from_nullspaces=False
    )
    dom_fs.set_alg_metrics(AlgMetrics())
    for path in expansion_order[1:-1]:
        cg.add_vertex_path_to_graph(path)
        node = SearchNode.from_vertex_path(path)
        AH_poly_ns, _ = dom_ns._create_path_AH_polytope_from_nullspace_sets(node)
        AH_poly_fs, _ = dom_fs._create_path_AH_polytope_from_full_sets(node)
        # They should be the same polyhedrons
        assert dom_fs.is_contained_in(AH_poly_ns, AH_poly_fs)
        assert dom_fs.is_contained_in(AH_poly_fs, AH_poly_ns)


def test_reaches_new_containment_cg_stackpush_d2():
    graph_file = ContactGraphGeneratorParams.inc_graph_file_path_from_name(
        "cg_stackpush_d2"
    )
    cg = IncrementalContactGraph.load_from_file(
        graph_file,
        should_incl_simul_mode_switches=False,
        should_add_const_edge_cost=True,
        should_add_gcs=True,
        should_use_l1_norm_vertex_cost=True,
    )
    domination_checker = ReachesNewContainment(graph=cg, containment_condition=-1)
    domination_checker.set_alg_metrics(AlgMetrics())

    # fmt: off
    candidate_vertex_path = ['source', "('NC|obs0_f3-obj0_v1', 'NC|obs0_f3-obj1_v1', 'NC|obs0_f3-obj2_v1', 'NC|obs0_f0-rob0_f2', 'NC|obj0_f1-obj1_f3', 'NC|obj0_f1-obj2_f3', 'NC|obj0_f0-rob0_f2', 'NC|obj1_f1-obj2_f3', 'NC|obj1_f0-rob0_f2', 'NC|obj2_f0-rob0_f2')", "('NC|obs0_f3-obj0_v1', 'NC|obs0_f3-obj1_v1', 'NC|obs0_f3-obj2_v1', 'NC|obs0_f3-rob0_v1', 'NC|obj0_f1-obj1_f3', 'NC|obj0_f1-obj2_f3', 'NC|obj0_f0-rob0_f2', 'NC|obj1_f1-obj2_f3', 'NC|obj1_f0-rob0_f2', 'NC|obj2_f0-rob0_f2')", "('NC|obs0_f3-obj0_v1', 'NC|obs0_f3-obj1_v1', 'NC|obs0_f3-obj2_v1', 'NC|obs0_f3-rob0_v1', 'NC|obj0_f1-obj1_f3', 'NC|obj0_f1-obj2_f3', 'IC|obj0_f3-rob0_f1', 'NC|obj1_f1-obj2_f3', 'NC|obj1_f0-rob0_f2', 'NC|obj2_f0-rob0_f2')", "('IC|obs0_f3-obj0_v1', 'NC|obs0_f3-obj1_v1', 'NC|obs0_f3-obj2_v1', 'NC|obs0_f3-rob0_v1', 'NC|obj0_f1-obj1_f3', 'NC|obj0_f1-obj2_f3', 'IC|obj0_f3-rob0_f1', 'NC|obj1_f1-obj2_f3', 'NC|obj1_f0-rob0_f2', 'NC|obj2_f0-rob0_f2')", "('IC|obs0_f3-obj0_v1', 'NC|obs0_f3-obj1_v1', 'NC|obs0_f3-obj2_v1', 'NC|obs0_f3-rob0_v1', 'IC|obj0_f1-obj1_f3', 'NC|obj0_f1-obj2_f3', 'IC|obj0_f3-rob0_f1', 'NC|obj1_f1-obj2_f3', 'NC|obj1_f0-rob0_f2', 'NC|obj2_f0-rob0_f2')", "('IC|obs0_f3-obj0_v1', 'IC|obs0_f3-obj1_v1', 'NC|obs0_f3-obj2_v1', 'NC|obs0_f3-rob0_v1', 'IC|obj0_f1-obj1_f3', 'NC|obj0_f1-obj2_f3', 'IC|obj0_f3-rob0_f1', 'NC|obj1_f1-obj2_f3', 'NC|obj1_f0-rob0_f2', 'NC|obj2_f0-rob0_f2')", "('IC|obs0_f3-obj0_v1', 'IC|obs0_f3-obj1_v1', 'NC|obs0_f3-obj2_v1', 'NC|obs0_f3-rob0_v1', 'IC|obj0_f1-obj1_f3', 'NC|obj0_f1-obj2_f3', 'IC|obj0_f3-rob0_f1', 'IC|obj1_f1-obj2_f3', 'NC|obj1_f0-rob0_f2', 'NC|obj2_f0-rob0_f2')", "('IC|obs0_f3-obj0_v1', 'IC|obs0_f3-obj1_v1', 'IC|obs0_f3-obj2_v1', 'NC|obs0_f3-rob0_v1', 'IC|obj0_f1-obj1_f3', 'NC|obj0_f1-obj2_f3', 'IC|obj0_f3-rob0_f1', 'IC|obj1_f1-obj2_f3', 'NC|obj1_f0-rob0_f2', 'NC|obj2_f0-rob0_f2')", "('IC|obs0_f3-obj0_v1', 'IC|obs0_f3-obj1_v1', 'IC|obs0_f2-obj2_f0', 'NC|obs0_f3-rob0_v1', 'IC|obj0_f1-obj1_f3', 'NC|obj0_f1-obj2_f3', 'IC|obj0_f3-rob0_f1', 'IC|obj1_f1-obj2_f3', 'NC|obj1_f0-rob0_f2', 'NC|obj2_f0-rob0_f2')", "('IC|obs0_f3-obj0_v1', 'IC|obs0_f2-obj1_f0', 'IC|obs0_f2-obj2_f0', 'NC|obs0_f3-rob0_v1', 'IC|obj0_f1-obj1_f3', 'NC|obj0_f1-obj2_f3', 'IC|obj0_f3-rob0_f1', 'IC|obj1_f1-obj2_f3', 'NC|obj1_f0-rob0_f2', 'NC|obj2_f0-rob0_f2')", "('IC|obs0_f3-obj0_v1', 'IC|obs0_f2-obj1_f0', 'NC|obs0_f1-obj2_f3', 'NC|obs0_f3-rob0_v1', 'IC|obj0_f1-obj1_f3', 'NC|obj0_f1-obj2_f3', 'IC|obj0_f3-rob0_f1', 'IC|obj1_f1-obj2_f3', 'NC|obj1_f0-rob0_f2', 'NC|obj2_f0-rob0_f2')", "('IC|obs0_f2-obj0_f0', 'IC|obs0_f2-obj1_f0', 'NC|obs0_f1-obj2_f3', 'NC|obs0_f3-rob0_v1', 'IC|obj0_f1-obj1_f3', 'NC|obj0_f1-obj2_f3', 'IC|obj0_f3-rob0_f1', 'IC|obj1_f1-obj2_f3', 'NC|obj1_f0-rob0_f2', 'NC|obj2_f0-rob0_f2')"]
    # fmt: on

    # Load these two paths into the graph
    cg.add_vertex_path_to_graph(candidate_vertex_path)

    # Check that subpaths of the candidate path are contained within themselves
    # for i in [len(candidate_vertex_path) - 1]:
    for i in range(2, len(candidate_vertex_path)):
        logger.debug(f"Checking subpath {i}")
        candidate_node = SearchNode.from_vertex_path(candidate_vertex_path[:i])
        assert domination_checker.is_dominated(
            candidate_node=candidate_node, alternate_nodes=[candidate_node]
        )
