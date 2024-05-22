from abc import ABC, abstractmethod

from large_gcs.algorithms.search_algorithm import AlgMetrics
from large_gcs.graph.graph import Edge, Graph, ShortestPathSolution


class CostEstimator(ABC):
    """Abstract base class for cost estimators that estimate the cost of a path
    from source to target through a particular edge."""

    def set_alg_metrics(self, alg_metrics: AlgMetrics):
        self._alg_metrics = alg_metrics

    def setup_subgraph(self, subgraph: Graph):
        """Setup the subgraph for use by the cost estimator."""

    @abstractmethod
    def estimate_cost(
        self, subgraph: Graph, edge: Edge, **kwargs
    ) -> ShortestPathSolution:
        """Estimate the cost of a path from source to target through a
        particular edge."""

    @property
    @abstractmethod
    def finger_print(self) -> str:
        """Return a string that uniquely identifies this cost estimator and
        it's relevant parameters."""
