import heapq as heap
import itertools
import logging
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from pydrake.all import Intersection
from tqdm import tqdm

from large_gcs.graph.graph import Edge, Graph

logger = logging.getLogger(__name__)

# Named tuple for the key in the vertices dictionary
LBGVertexKey = namedtuple("LBGVertexKey", ["parent_triple", "parent_vertex"])


@dataclass
class LBGVertex:
    """Each vertex is uniquely defined by the (predecessor, vertex, successor)
    and parent vertex (predecessor or successor) in the original GCS.

    The point is the terminal point in the successor of the convex
    restriction over those two edges.
    """

    parent_triple: Tuple[str, str, str]
    parent_vertex: str
    point: np.ndarray

    @property
    def key(self):
        return (self.parent_triple, self.parent_vertex)


class LowerBoundGraph:
    def __init__(self, graph: Graph):
        # Each vertex is the terminal
        self.vertices: Dict[LBGVertexKey, LBGVertex] = {}
        self.edges: Dict[Tuple[LBGVertexKey, LBGVertexKey], float] = {}
        self._graph = graph

        logger.debug("Processing vertices of original graph...")
        for vertex in tqdm(self._graph.vertices, desc="Vertices"):
            logger.debug("Findings solutions for triplets...")
            edge_pairs = list(
                itertools.product(
                    self._graph.incoming_edges(vertex),
                    self._graph.outgoing_edges(vertex),
                )
            )
            for incoming_edge, outgoing_edge in tqdm(edge_pairs, desc="Edge pairs"):
                predecessor = incoming_edge.u
                successor = outgoing_edge.v
                if predecessor == successor:
                    continue

                self._graph.set_source(predecessor)
                self._graph.set_target(successor)
                sol = self._graph.solve_convex_restriction(
                    active_edge_keys=[incoming_edge.key, outgoing_edge.key],
                    skip_post_solve=False,
                )
                if not sol.is_success:
                    continue

                lbg_vertex_pred = LBGVertex(
                    parent_triple=(predecessor, vertex, successor),
                    parent_vertex=predecessor,
                    point=sol.ambient_path[0],
                )
                lbg_vertex_succ = LBGVertex(
                    parent_triple=(predecessor, vertex, successor),
                    parent_vertex=successor,
                    point=sol.ambient_path[-1],
                )
                self.add_vertex(lbg_vertex_pred)
                self.add_vertex(lbg_vertex_succ)
                self.add_edge(lbg_vertex_pred.key, lbg_vertex_succ.key, sol.cost)
                self.add_edge(lbg_vertex_succ.key, lbg_vertex_pred.key, sol.cost)
        logger.debug("Processing edges of original graph...")
        for edge in tqdm(self._graph.edges, desc="Edges"):
            u = self._graph.edges[edge].u
            v = self._graph.edges[edge].v
            u_set = self._graph.vertices[u].convex_set.set
            v_set = self._graph.vertices[v].convex_set.set
            intersection = Intersection(u_set, v_set)
            vertex_pairs = list(itertools.combinations(self.vertices.values(), 2))
            logger.debug(f"Checking intersection of {u} and {v}")
            for q1, q2 in tqdm(vertex_pairs, desc="LBG Vertex pairs"):
                logger.debug(f"q1 {len(q1.point)} ambient dim")
                if intersection.PointInSet(q1.point) and intersection.PointInSet(
                    q2.point
                ):
                    self.add_edge(q1.key, q2.key, 0)
                    self.add_edge(q2.key, q1.key, 0)

    def add_vertex(self, LBG_vertex: LBGVertex):
        self.vertices[LBG_vertex.key] = LBG_vertex

    def add_edge(self, u: LBGVertexKey, v: LBGVertexKey, cost: float):
        self.edges[(u, v)] = cost

    def outgoing_edges(self, v: str) -> List[Tuple[str, str]]:
        """Get the outgoing edges of a vertex."""
        assert v in self.vertices
        return [edge for edge in self.edges.values() if edge.u == v]

    def incoming_edges(self, vertex_name: str) -> List[Tuple[str, str]]:
        """Get the incoming edges of a vertex."""
        assert vertex_name in self.vertices
        return [edge for edge in self.edges.values() if edge.v == vertex_name]

    def run_dijkstra(self, source: str) -> Tuple[float, List[str]]:
        self._g = {vertex: float("inf") for vertex in self.vertices}
        self._g[source] = 0
        expanded = set()
        Q = []
        source_vertex = (0, source, [])
        heap.heappush(Q, source_vertex)
        while len(Q) > 0:
            cost, vertex_name, path = heap.heappop(Q)

            if vertex_name in expanded:
                continue
            expanded.add(vertex_name)
            for u, neighbor in self.outgoing_edges(vertex_name):
                if neighbor in expanded:
                    continue

                edge_cost = self.edges[(vertex_name, neighbor)]
                if self._g[neighbor] > self._g[vertex_name] + edge_cost:
                    self._g[neighbor] = self._g[vertex_name] + edge_cost
                    heap.heappush(
                        Q, (self._g[neighbor], neighbor, path + [vertex_name])
                    )
        logger.debug(f"Finished Dijkstra")

    def get_cost_to_go(self, vertex: str) -> float:
        return self._g[vertex]
