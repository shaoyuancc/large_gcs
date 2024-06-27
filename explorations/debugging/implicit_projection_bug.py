import logging

from large_gcs.algorithms.gcs_star import GcsStar
from large_gcs.cost_estimators.shortcut_edge_ce import ShortcutEdgeCE
from large_gcs.graph.contact_cost_constraint_factory import (
    contact_shortcut_edge_l2norm_cost_factory_obj_weighted,
)

logging.basicConfig(level=logging.WARN)
logging.getLogger("large_gcs").setLevel(logging.DEBUG)

from large_gcs.graph.graph import ShortestPathSolution
from large_gcs.graph.incremental_contact_graph import IncrementalContactGraph
from large_gcs.graph_generators.contact_graph_generator import (
    ContactGraphGeneratorParams,
)

# cg_maze_b1_1 cg_maze_b1
graph_file = ContactGraphGeneratorParams.inc_graph_file_path_from_name("cg_maze_b1")
cg = IncrementalContactGraph.load_from_file(
    graph_file,
    should_incl_simul_mode_switches=False,
    should_add_const_edge_cost=True,
    should_add_gcs=True,
)
cost_estimator = ShortcutEdgeCE(
    cg,
    shortcut_edge_cost_factory=contact_shortcut_edge_l2norm_cost_factory_obj_weighted,
)
alg = GcsStar(cg, cost_estimator, num_samples_per_vertex=5)
sol: ShortestPathSolution = alg.run()
