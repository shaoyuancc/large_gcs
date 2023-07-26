from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from enum import Enum
from math import inf

import numpy as np


class TieBreak(Enum):
    FIFO = 1
    LIFO = 2


@dataclass
class AlgVisParams:
    """
    Parameters for visualizing the algorithself.
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
    Metrics for the algorithself.
    """

    n_vertices_visited: int = 0
    # Note that this is not the number of edges relaxed. It is the number of edges in the visited subgraph.
    n_edges_visited: int = 0
    n_vertices_explored: int = 0
    vertex_coverage: float = 0
    edge_coverage: float = 0
    time_wall_clock: float = 0.0
    n_gcs_solves: int = 0
    gcs_solve_time_total: float = 0.0
    gcs_solve_time_iter_mean: float = 0.0
    gcs_solve_time_last_10_mean: float = 0.0
    gcs_solve_time_iter_std: float = 0.0
    gcs_solve_time_iter_min: float = inf
    gcs_solve_time_iter_max: float = 0.0
    n_vertices_revisited: int = 0
    n_vertices_reexplored: int = 0

    def __post_init__(self):
        self._gcs_solve_times = np.empty((0,))

    def update_after_gcs_solve(self, solve_time: float):
        self.n_gcs_solves += 1
        self.gcs_solve_time_total += solve_time

        if solve_time < self.gcs_solve_time_iter_min:
            self.gcs_solve_time_iter_min = solve_time
        if solve_time > self.gcs_solve_time_iter_max:
            self.gcs_solve_time_iter_max = solve_time
        self._gcs_solve_times = np.append(self._gcs_solve_times, solve_time)

    def update_derived_metrics(
        self,
        n_vertices,
        n_edges,
        n_visited_edges,
    ):
        """Recompute metrics based on the current state of the algorithself.
        n_vertices_visited, n_gcs_solves, gcs_solve_time_total/min/max are manually updated.
        The rest are computed from the manually updated metrics.
        """
        self.vertex_coverage = round(self.n_vertices_visited / n_vertices, 4)
        self.n_edges_visited = n_visited_edges
        self.edge_coverage = round(self.n_edges_visited / n_edges, 4)
        if self.n_gcs_solves > 0:
            self.gcs_solve_time_iter_mean = (
                self.gcs_solve_time_total / self.n_gcs_solves
            )
            self.gcs_solve_time_iter_std = np.std(self._gcs_solve_times)
        if self.n_gcs_solves > 10:
            self.gcs_solve_time_last_10_mean = np.mean(self._gcs_solve_times[-10:])
        return self

    def __str__(self):
        result = []
        for field in fields(self):
            # Skip private fields
            if field.name.startswith("_"):
                continue
            value = getattr(self, field.name)
            if isinstance(value, float):
                # Format the float to 3 significant figures
                value = "{:.3g}".format(value)
            result.append(f"{field.name}: {value}")
        return ", ".join(result)

    def to_dict(self):
        """Hide private fields and return a dictionary of the public fields."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if not f.name.startswith("_")
        }


class SearchAlgorithm(ABC):
    """
    Abstract base class for search algorithms.
    """

    @abstractmethod
    def run(self):
        """
        Searches for a shortest path in the given graph.
        """
