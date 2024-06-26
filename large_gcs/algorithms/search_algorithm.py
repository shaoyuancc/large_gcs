import heapq as heap
import itertools
import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, fields
from enum import Enum
from functools import wraps
from math import inf
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

import wandb
from large_gcs.graph.graph import Edge, Graph, ShortestPathSolution
from large_gcs.utils.utils import dict_to_dataclass

logger = logging.getLogger(__name__)


def profile_method(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = method(self, *args, **kwargs)  # Call the original method
        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time > 300:
            logger.warning(
                f"Method {method.__name__} took {elapsed_time:.2f} seconds to run."
            )
        # Update the AlgMetrics with the elapsed time for the method
        self._alg_metrics.method_times[method.__name__] += elapsed_time
        self._alg_metrics.method_counts[method.__name__] += 1

        return result

    return wrapper


class TieBreak(Enum):
    FIFO = 1
    LIFO = 2


class ReexploreLevel(Enum):
    NONE = 0
    PARTIAL = 1
    FULL = 2


@dataclass
class AlgVisParams:
    """Parameters for visualizing the algorithm."""

    log_dir: Optional[str] = None
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
    """Metrics for the algorithm."""

    n_vertices_expanded: int = 0
    n_vertices_visited: int = 0
    time_wall_clock: float = 0.0
    n_gcs_solves: int = 0
    gcs_solve_time_total: float = 0.0
    gcs_solve_time_iter_mean: float = 0.0
    gcs_solve_time_last_10_mean: float = 0.0
    gcs_solve_time_iter_std: float = 0.0
    gcs_solve_time_iter_min: float = inf
    gcs_solve_time_iter_max: float = 0.0
    n_vertices_reexpanded: int = 0
    n_vertices_revisited: int = 0
    n_Q: int = 0
    n_S: int = 0
    n_S_pruned: int = 0
    _S_pruned_counts: DefaultDict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    method_times: DefaultDict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    method_counts: DefaultDict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    expansion_order: List[str] = field(default_factory=list)

    def __post_init__(self):
        self._gcs_solve_times = np.empty((0,))
        self._method_call_structure = None

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
    ):
        """Recompute metrics based on the current state of the algorithself.

        n_vertices_visited, n_gcs_solves, gcs_solve_time_total/min/max
        are manually updated. The rest are computed from the manually
        updated metrics.
        """
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
            elif isinstance(value, defaultdict):
                # Convert defaultdicts to dict for printing
                value = dict(value)
            result.append(f"{field.name}: {value}")

        return ", ".join(result)

    def to_dict(self):
        """Hide private fields and return a dictionary of the public fields."""
        res = {}
        for f in fields(self):
            # Skip private fields
            if f.name.startswith("_"):
                continue
            attr = getattr(self, f.name)
            if isinstance(attr, np.ndarray):
                res[f.name] = attr.tolist()  # Convert numpy arrays to lists
            elif isinstance(attr, defaultdict):
                res[f.name] = dict(attr)  # Convert defaultdict to dict
            if isinstance(attr, dict):
                # Convert the keys to strings if they are ints (it's creating problems for wandb)
                res[f.name] = {
                    (str(k) if isinstance(k, int) else k): v for k, v in attr.items()
                }
            else:
                res[f.name] = attr
        return res

    def update_method_call_structure(self, call_structure: Dict[str, List[str]]):
        """Set the call structure of the methods for the method time pie
        chart."""
        if self._method_call_structure is not None:
            for key, value in call_structure.items():
                if key in self._method_call_structure:
                    self._method_call_structure[key].extend(value)
                else:
                    self._method_call_structure[key] = value
        else:
            self._method_call_structure = call_structure

        # Note that this calculation will be wrong if a child method is called by two parents
        # But I disabled this check because I have a child method called by two parents but not in the same run (i.e. with different settings)
        # called_methods = set()
        # for nested_methods in call_structure.values():
        #     for nested_method in nested_methods:
        #         assert (
        #             nested_method not in called_methods
        #         ), f"Method {nested_method} is called by multiple parent methods."
        #         called_methods.add(nested_method)

    @property
    def method_call_structure(self):
        return self._method_call_structure

    def generate_method_time_piechart(self):
        """Generate a pie chart of the time spent in each method."""
        assert (
            self._method_call_structure is not None
        ), "Method call structure must be set before generating the pie chart."

        # Calculate exclusive times from total times that include nested calls
        exclusive_times = {method: times for method, times in self.method_times.items()}
        for method, nested_methods in self._method_call_structure.items():
            if not method in exclusive_times:
                continue
            for nested_method in nested_methods:
                exclusive_times[method] -= self.method_times[nested_method]

        # This may be called "mid-nest" so we need to remove the negative values
        exclusive_times = {
            method: max(0, time) for method, time in exclusive_times.items()
        }

        # Generate the pie chart
        labels = list(exclusive_times.keys())
        values = list(exclusive_times.values())

        # Create a pie chart
        fig = go.Figure()
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                pull=[0.1] * len(labels),
                textinfo="label+value+percent",
            )
        )

        # Enhance chart visuals
        fig.update_traces(marker=dict(line=dict(color="#000000", width=2)))
        fig.update_layout(title_text="Exclusive Method Execution Times", title_x=0.5)
        return fig

    def generate_tracked_pruned_paths_histogram(
        self, tracked_counts: List[int], pruned_counts: List[int]
    ):
        # Create a figure to plot the histograms
        fig = go.Figure()

        # Adding Tracked histogram
        fig.add_trace(go.Histogram(x=tracked_counts, name="Tracked", opacity=0.75))

        # Adding Pruned histogram
        fig.add_trace(go.Histogram(x=pruned_counts, name="Pruned", opacity=0.75))

        # Update layout for a stacked or overlaid histogram
        fig.update_layout(
            title_text="Tracked and Pruned Paths per Vertex Histogram",  # Title
            xaxis_title_text="Number of Paths to Vertex",  # x-axis label
            yaxis_title_text="Frequency",  # y-axis label
            barmode="stack",
        )
        fig.update_traces(marker_line_width=1.5)

        return fig

    def save(self, loc: Path) -> None:
        """Save metrics as a JSON file."""
        # Convert to dictionary and save to JSON
        self_as_dict = self.to_dict()
        import json

        with open(loc, "w") as f:
            json.dump(self_as_dict, f, indent=4)

    @classmethod
    def load(cls, loc: Path) -> "AlgMetrics":
        """Reads metrics from a JSON file."""
        # Load from JSON and convert back to dataclass
        import json

        with open(loc, "r") as f:
            loaded_dict = json.load(f)

        metrics = dict_to_dataclass(AlgMetrics, loaded_dict)
        return metrics


@dataclass
class SearchNode:
    """A node in the search tree."""

    priority: float
    vertex_name: str
    # Edge path
    edge_path: List[str]
    # Vertex path
    vertex_path: List[str]
    parent: Optional["SearchNode"] = None
    sol: Optional[ShortestPathSolution] = None
    ah_polyhedron_ns: Optional["AH_polytope"] = None  # type: ignore
    ah_polyhedron_fs: Optional["AH_polytope"] = None  # type: ignore

    def __lt__(self, other: "SearchNode"):
        return self.priority < other.priority

    @classmethod
    def from_source(cls, source_vertex_name: str):
        return cls(
            priority=0,
            vertex_name=source_vertex_name,
            edge_path=[],
            vertex_path=[source_vertex_name],
        )

    @classmethod
    def from_parent(cls, child_vertex_name: str, parent: "SearchNode"):
        new_edge = Edge(u=parent.vertex_name, v=child_vertex_name)
        return cls(
            priority=None,
            vertex_name=child_vertex_name,
            edge_path=parent.edge_path.copy() + [new_edge.key],
            vertex_path=parent.vertex_path.copy() + [child_vertex_name],
            parent=parent,
        )

    @classmethod
    def from_vertex_path(cls, vertex_path: List[str]):
        edge_path = []
        for u, v in zip(vertex_path[:-1], vertex_path[1:]):
            # Note that this will break if the edges in the graph have key_suffixes
            edge_path.append(Edge(u, v).key)
        return cls(
            priority=None,
            vertex_name=vertex_path[-1],
            edge_path=edge_path,
            vertex_path=vertex_path,
            parent=None,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sol"] = None  # Do not serialize `sol`
        return state


class SearchAlgorithm(ABC):
    """Abstract base class for search algorithms."""

    def __init__(
        self,
        graph: Graph,
        vis_params: Optional[AlgVisParams] = None,
        heuristic_inflation_factor: float = 1.0,
        tiebreak: TieBreak = TieBreak.FIFO,
    ):
        self._graph = graph
        self._target_name = graph.target_name
        self._heuristic_inflation_factor = heuristic_inflation_factor

        # Visited dictionary
        self._S: DefaultDict[str, deque[SearchNode]] = defaultdict(deque)
        # Priority queue
        self._Q = []

        self._tiebreak = tiebreak
        if tiebreak == TieBreak.FIFO or tiebreak == TieBreak.FIFO.name:
            self._counter = itertools.count(start=0, step=1)
        elif tiebreak == TieBreak.LIFO or tiebreak == TieBreak.LIFO.name:
            self._counter = itertools.count(start=0, step=-1)

        # For logging/metrics
        self._vis_params = vis_params
        self._alg_metrics = AlgMetrics()
        self._expanded = set()
        self._visited = set()
        self._last_plots_save_time = time.time()
        self._step = 0

    @abstractmethod
    def run(self):
        """Searches for a shortest path in the given graph."""

    def push_node_on_Q(self, node: SearchNode):
        # Abstraction for the priority queue push operation that handles tiebreaks
        heap.heappush(self._Q, (node, next(self._counter)))

    def pop_node_from_Q(self):
        # Abstraction for the priority queue pop operation that handles tiebreaks
        return heap.heappop(self._Q)[0]

    def set_node_in_S(self, n: SearchNode):
        self._S[n.vertex_name] = deque([n])

    def add_node_to_S_left(self, n: SearchNode):
        self._S[n.vertex_name].appendleft(n)

    def add_node_to_S(self, n: SearchNode):
        self._S[n.vertex_name].append(n)

    def remove_node_from_S_left(self, vertex_name: str):
        self._S[vertex_name].pop()

    def remove_node_from_S(self, vertex_name: str):
        self._S[vertex_name].popleft()

    def update_expanded(self, n: SearchNode):
        if n.vertex_name in self._expanded:
            self._alg_metrics.n_vertices_reexpanded += 1
        else:
            self._expanded.add(n.vertex_name)
            self._alg_metrics.n_vertices_expanded += 1

    def update_visited(self, n: SearchNode):
        # Purely for logging purposes
        if n.vertex_name in self._visited:
            self._alg_metrics.n_vertices_revisited += 1
        else:
            self._alg_metrics.n_vertices_visited += 1
            self._visited.add(n.vertex_name)

    def update_pruned(self, n: SearchNode):
        self._alg_metrics._S_pruned_counts[n.vertex_name] += 1

    @property
    def alg_metrics(self):
        self._alg_metrics.n_Q = len(self._Q)
        self._alg_metrics.n_S = sum(len(lst) for lst in self._S.values())
        self._alg_metrics.n_S_pruned = sum(self._alg_metrics._S_pruned_counts.values())
        return self._alg_metrics.update_derived_metrics()

    def log_metrics_to_wandb(self, total_estimated_cost: float):
        if wandb.run is not None:
            wandb.log(
                {
                    "total_estimated_cost": total_estimated_cost,
                    "alg_metrics": self.alg_metrics.to_dict(),
                }
            )

    def save_alg_metrics_to_file(self, loc: Path) -> None:
        self._alg_metrics.save(loc)

    @profile_method
    def _save_metrics(self, n: SearchNode, n_successors: int, override_save=False):
        logger.info(
            f"iter: {self._step}\n{self.alg_metrics}\nnow exploring node {n.vertex_name}'s {n_successors} neighbors ({n.priority})"
        )
        self._step += 1
        current_time = time.time()
        PERIOD = 1200

        if self._vis_params is not None and (
            override_save or self._last_plots_save_time + PERIOD < current_time
        ):
            log_dir = self._vis_params.log_dir
            # Histogram of paths per vertex
            # Preparing tracked and pruned counts
            tracked_counts = [len(self._S[v]) for v in self._S]
            pruned_counts = [
                self._alg_metrics._S_pruned_counts[v]
                for v in self._alg_metrics._S_pruned_counts
            ]
            log_dict = {}
            hist_fig = self.alg_metrics.generate_tracked_pruned_paths_histogram(
                tracked_counts, pruned_counts
            )
            # Save the figure to a file as png
            hist_fig.write_image(os.path.join(log_dir, "paths_per_vertex_hist.png"))
            log_dict["paths_per_vertex_hist"] = wandb.Plotly(hist_fig)

            # Pie chart of method times
            pie_fig = self._alg_metrics.generate_method_time_piechart()
            pie_fig.write_image(os.path.join(log_dir, "method_times_pie_chart.png"))
            log_dict["method_times_pie_chart"] = wandb.Plotly(pie_fig)

            checkpoint_path_str = self.save_checkpoint()

            if wandb.run is not None:
                # Log the Plotly figure and other metrics to wandb
                wandb.log(log_dict, step=self._step)
                wandb.save(checkpoint_path_str)

            self._last_plots_save_time = current_time
        self.log_metrics_to_wandb(n.priority)

    def save_checkpoint(self):
        logger.info("Saving checkpoint")
        log_dir = self._vis_params.log_dir
        file_path = os.path.join(log_dir, "checkpoint.pkl")
        checkpoint_data = {
            "Q": self._Q,
            "S": self._S,
            "expanded": self._expanded,
            "visited": self._visited,
            "counter": next(self._counter) - 1,
            "step": self._step,
            "alg_metrics": self._alg_metrics,
        }
        with open(file_path, "wb") as f:
            pickle.dump(checkpoint_data, f)
        return str(file_path)

    def load_checkpoint(self, checkpoint_log_dir, override_wall_clock_time=None):
        file_path = os.path.join(checkpoint_log_dir, "checkpoint.pkl")
        with open(file_path, "rb") as f:
            checkpoint_data = pickle.load(f)
        self._Q = checkpoint_data["Q"]
        self._S = checkpoint_data["S"]
        self._expanded = checkpoint_data["expanded"]
        self._visited = checkpoint_data["visited"]

        if self._tiebreak == TieBreak.FIFO or self._tiebreak == TieBreak.FIFO.name:
            self._counter = itertools.count(start=checkpoint_data["counter"], step=1)
        elif self._tiebreak == TieBreak.LIFO or self._tiebreak == TieBreak.LIFO.name:
            self._counter = itertools.count(start=checkpoint_data["counter"], step=-1)
        self._step = checkpoint_data["step"]
        self._alg_metrics: AlgMetrics = checkpoint_data["alg_metrics"]
        if override_wall_clock_time is not None:
            self._alg_metrics.time_wall_clock = override_wall_clock_time

    def log_metrics_to_wandb(self, total_estimated_cost: float):
        if wandb.run is not None:  # not self._S
            wandb.log(
                {
                    "total_estimated_cost": total_estimated_cost,
                    "alg_metrics": self.alg_metrics.to_dict(),
                },
                self._step,
            )
