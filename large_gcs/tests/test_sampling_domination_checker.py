from large_gcs.algorithms.search_algorithm import AlgMetrics, SearchNode
from large_gcs.domination_checkers.reaches_cheaper_sampling import (
    ReachesCheaperSampling,
)
from large_gcs.domination_checkers.reaches_cheaper_sampling_pairwise import (
    ReachesCheaperSamplingPairwise,
)
from large_gcs.domination_checkers.reaches_new_sampling import ReachesNewSampling
from large_gcs.domination_checkers.reaches_new_sampling_pairwise import (
    ReachesNewSamplingPairwise,
)
from large_gcs.graph_generators.hor_vert_gcs import create_polyhedral_hor_vert_b_graph

NUM_SAMPLES_PER_VERTEX = 200


def test_reaches_new_sampling_pairwise_polyhedral_hor_vert_b_graph():
    G = create_polyhedral_hor_vert_b_graph()
    domination_checker = ReachesNewSamplingPairwise(
        graph=G, num_samples_per_vertex=NUM_SAMPLES_PER_VERTEX
    )
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


def test_reaches_cheaper_sampling_pairwise_polyhedral_hor_vert_b_graph():
    G = create_polyhedral_hor_vert_b_graph()
    domination_checker = ReachesCheaperSamplingPairwise(
        graph=G, num_samples_per_vertex=NUM_SAMPLES_PER_VERTEX
    )
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


def test_reaches_new_sampling_polyhedral_hor_vert_b_graph():
    G = create_polyhedral_hor_vert_b_graph()
    domination_checker = ReachesNewSampling(
        graph=G, num_samples_per_vertex=NUM_SAMPLES_PER_VERTEX
    )
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
    n_a = SearchNode.from_vertex_path(["s", "p0", "p2"])
    n_b = SearchNode.from_vertex_path(["s", "p3", "p2"])
    n_c = SearchNode.from_vertex_path(["s", "p4", "p2"])
    n_d = SearchNode.from_vertex_path(["s", "p5", "p2"])
    assert domination_checker.is_dominated(
        candidate_node=n_a, alternate_nodes=[n_b, n_c, n_d]
    )


def test_reaches_cheaper_sampling_polyhedral_hor_vert_b_graph():
    G = create_polyhedral_hor_vert_b_graph()
    domination_checker = ReachesCheaperSampling(
        graph=G, num_samples_per_vertex=NUM_SAMPLES_PER_VERTEX
    )
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
    n_a = SearchNode.from_vertex_path(["s", "p0", "p2"])
    n_b = SearchNode.from_vertex_path(["s", "p3", "p2"])
    n_c = SearchNode.from_vertex_path(["s", "p4", "p2"])
    n_d = SearchNode.from_vertex_path(["s", "p5", "p2"])
    assert domination_checker.is_dominated(
        candidate_node=n_a, alternate_nodes=[n_b, n_c, n_d]
    )
