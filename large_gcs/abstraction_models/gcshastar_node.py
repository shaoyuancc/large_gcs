from dataclasses import dataclass
from typing import List, Optional

from large_gcs.graph.graph import ShortestPathSolution


@dataclass
class GCSHANode:
    priority: Optional[float]
    abs_level: int
    vertex_name: str
    path: list
    weight: Optional[float]
    parent: Optional["GCSHANode"] = None
    sol: Optional[ShortestPathSolution] = None

    @property
    def id(self):
        return f"{self.abs_level}_{self.__class__.__name__}_{self.vertex_name}"

    def __lt__(self, other: "GCSHANode"):
        return self.priority < other.priority


@dataclass
class StatementNode(GCSHANode):
    @classmethod
    def from_parent(cls, child_vertex_name: str, parent: "GCSHANode"):
        assert isinstance(parent, StatementNode)
        return cls(
            priority=None,
            abs_level=parent.abs_level,
            vertex_name=child_vertex_name,
            path=parent.path + [(parent.vertex_name, child_vertex_name)],
            weight=None,
            parent=parent,
        )

    @property
    def context_id(self) -> str:
        return f"{self.abs_level}_{ContextNode.__name__}_{self.vertex_name}"


@dataclass
class ContextNode(GCSHANode):
    # Each entry corresponds to vertex_cost(u) + edge_cost(u, v) for (u,v) in path that shares the same index
    path_costs: List[float] = None
