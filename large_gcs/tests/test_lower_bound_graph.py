import logging

from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.lower_bound_graph import LowerBoundGraph
from large_gcs.graph_generators.contact_graph_generator import (
    ContactGraphGeneratorParams,
)

logging.basicConfig(level=logging.WARN)
logging.getLogger("large_gcs").setLevel(logging.DEBUG)
logging.getLogger("large_gcs.geometry.convex_set").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def test_lbg_vertices_w_target_parent_are_0_cost():
    graph_file = ContactGraphGeneratorParams.graph_file_path_from_name("cg_simple_1_1")
    cg = ContactGraph.load_from_file(
        graph_file,
        should_use_l1_norm_vertex_cost=True,
    )
    # cg.plot()
    lbg = LowerBoundGraph(cg)
    lbg.run_dijkstra()
    for vertex in lbg._parent_vertex_to_vertices[lbg._target_name]:
        assert lbg._g[vertex.key] == 0
