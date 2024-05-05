from large_gcs.algorithms.search_algorithm import SearchNode
from large_gcs.domination_checkers.reaches_cheaper_containment import (
    ReachesCheaperContainment,
)
from large_gcs.domination_checkers.reaches_new_containment import ReachesNewContainment
from large_gcs.graph_generators.hor_vert_gcs import create_polyhedral_hor_vert_b_graph
from large_gcs.graph_generators.one_dimensional_gcs import create_simple_1d_graph


def test_reaches_new_containment_polyhedral_hor_vert_b_graph():
    G = create_polyhedral_hor_vert_b_graph()
    domination_checker = ReachesNewContainment(graph=G, containment_condition=-1)
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
    n = SearchNode.from_vertex_path(["s", "t"])
    n_prime = SearchNode.from_vertex_path(["p0", "t"])

    assert domination_checker.is_dominated(candidate_node=n, alternate_nodes=[n_prime])
    assert not domination_checker.is_dominated(
        candidate_node=n_prime, alternate_nodes=[n]
    )
