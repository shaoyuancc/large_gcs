from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import inf


@dataclass
class AlgVisParams:
    """
    Parameters for visualizing the algorithm.
    """

    vid_output_path: str = "alg_vis_output.mp4"
    plot_output_path: str = "alg_vis_output.png"
    figsize: tuple = (5, 5)
    fps: int = 3
    dpi: int = 200
    visited_vertex_color: str = "lightskyblue"
    visited_edge_color: str = "lightseagreen"
    frontier_color: str = "lightyellow"
    relaxing_to_color: str = "lightgreen"
    relaxing_from_color: str = "skyblue"
    edge_color: str = "gray"
    relaxing_edge_color: str = "lime"
    intermediate_path_color: str = "lightgrey"
    final_path_color: str = "orange"


@dataclass
class AlgMetrics:
    """
    Metrics for the algorithm.
    """

    n_vertices_visited: int = 0
    # Note that this is not the number of edges relaxed. It is the number of edges in the visited subgraph.
    n_edges_visited: int = 0
    vertex_coverage: float = 0.0
    edge_coverage: float = 0.0
    n_gcs_solves: int = 0
    gcs_solve_time_total: float = 0.0
    gcs_solve_time_iter_mean: float = 0.0
    gcs_solve_time_iter_std: float = 0.0
    gcs_solve_time_iter_min: float = inf
    gcs_solve_time_iter_max: float = 0.0


class SearchAlgorithm(ABC):
    """
    Abstract base class for search algorithms.
    """

    @abstractmethod
    def run(self):
        """
        Searches for a shortest path in the given graph.
        """
        pass
