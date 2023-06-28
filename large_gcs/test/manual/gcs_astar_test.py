import os
import matplotlib.pyplot as plt
from large_gcs.algorithms.gcs_astar import GcsAstar
from large_gcs.algorithms.search_algorithm import AlgVisParams
from large_gcs.graph.cost_factory import create_l2norm_edge_cost
from large_gcs.example_graphs.utils.spp_shape_gcs_utils import load_spp_shape_gcs


base_file_name = "spp_2d_v100_t2_shape_gcs"
graph_file = os.path.join(
    os.environ["PROJECT_ROOT"], "large_gcs", "example_graphs", base_file_name + ".npy"
)
G = load_spp_shape_gcs(graph_file, create_l2norm_edge_cost)

output_dir = os.path.join(os.environ["PROJECT_ROOT"], "output", "gcs_astar")
vid_file = os.path.join(output_dir, f"gcs_astar_{base_file_name}_l2norm.mp4")
plot_file = os.path.join(output_dir, f"gcs_astar_{base_file_name}_l2norm.png")
vis_params = AlgVisParams(
    vid_output_path=vid_file, fps=10, figsize=(20, 20), plot_output_path=plot_file
)
gcs_astar = GcsAstar(G, vis_params).run(animate=False, verbose=False, final_plot=False)
