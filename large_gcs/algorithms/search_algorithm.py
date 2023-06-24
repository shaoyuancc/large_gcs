from abc import ABC, abstractmethod


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
