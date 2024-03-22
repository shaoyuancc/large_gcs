from dataclasses import dataclass
from typing import List, Optional

from large_gcs.graph.graph import Edge, ShortestPathSolution


@dataclass
class GCSHANode:
    priority: Optional[float]
    abs_level: int
    vertex_name: str
    edge_path: List[str]
    vertex_path: List[str]
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
        new_edge = Edge(u=parent.vertex_name, v=child_vertex_name)
        return cls(
            priority=None,
            abs_level=parent.abs_level,
            vertex_name=child_vertex_name,
            edge_path=parent.edge_path.copy() + [new_edge.key],
            vertex_path=parent.vertex_path.copy() + [child_vertex_name],
            weight=None,
            parent=parent,
        )

    @classmethod
    def create_start_node(
        cls, vertex_name: str, abs_level: int, priority: Optional[float] = None
    ):
        return cls(
            priority=priority,
            abs_level=abs_level,
            vertex_name=vertex_name,
            edge_path=[],
            vertex_path=[vertex_name],
            weight=None,
            parent=None,
        )

    @property
    def context_id(self) -> str:
        return f"{self.abs_level}_{ContextNode.__name__}_{self.vertex_name}"


@dataclass
class ContextNode(GCSHANode):
    # Each entry corresponds to vertex_cost(u) + edge_cost(u, v) for (u,v) in path that shares the same index
    path_costs: List[float] = None
