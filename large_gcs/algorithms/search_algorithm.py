from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AlgVisParams:
    """
    Parameters for visualizing the algorithm.
    """

    output_path: str = "alg_vis_output.mp4"
    fps: int = 2
    dpi: int = 200
    visited_color: str = "lightskyblue"
    frontier_color: str = "lightyellow"
    relaxing_to_color: str = "lightgreen"
    relaxing_from_color: str = "skyblue"
    edge_color: str = "gray"
    relaxing_edge_color: str = "lime"
    intermediate_path_color: str = "lightgrey"
    final_path_color: str = "orange"


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
