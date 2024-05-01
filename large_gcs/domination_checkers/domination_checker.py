from abc import ABC, abstractmethod
from typing import List

from large_gcs.algorithms.search_algorithm import AlgMetrics, SearchNode
from large_gcs.graph.graph import Graph


class DominationChecker(ABC):
    def __init__(self, graph: Graph) -> None:
        self._graph = graph
        self._target = graph.target_name

    @abstractmethod
    def is_dominated(
        self, candidate_node: SearchNode, alternate_nodes: List[SearchNode]
    ) -> bool:
        pass

    def set_alg_metrics(self, alg_metrics: AlgMetrics):
        self._alg_metrics = alg_metrics
