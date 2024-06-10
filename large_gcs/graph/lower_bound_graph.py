import heapq as heap
import itertools
import logging
import os
import time
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from large_gcs.graph.graph import Edge, Graph, Vertex

logger = logging.getLogger(__name__)

# Named tuple for the key in the vertices dictionary
LBGVertexKey = namedtuple("LBGVertexKey", ["pred", "vertex", "succ"])


@dataclass
class LBGVertex:
    """Each vertex is uniquely defined by the (predecessor, vertex, successor)
    in the original GCS.

    The point is the terminal point in the successor of the convex
    restriction over those two edges.
    """

    parent_triple: Tuple[str, str, str]
    point: np.ndarray

    @property
    def key(self):
        return self.parent_triple

    @property
    def parent_edge(self):
        return Edge.key_from_uv(self.parent_triple[1], self.parent_triple[2])

    @property
    def parent_vertex(self):
        return self.parent_triple[2]


class LowerBoundGraph:
    def __init__(self, graph_name: str, source_name: str, target_name: str):
        # Each vertex is the terminal
        self._vertices: Dict[LBGVertexKey, LBGVertex] = {}
        self._edges: Dict[Tuple[LBGVertexKey, LBGVertexKey], float] = {}
        self._graph: Optional[Graph] = None
        self._graph_name = graph_name
        self._source_name = source_name
        self._target_name = target_name
        self._parent_vertex_to_vertices: Dict[str, List[LBGVertexKey]] = defaultdict(
            list
        )
        self._parent_edge_to_vertices: Dict[str, List[LBGVertexKey]] = defaultdict(list)
        self._g: Dict[LBGVertexKey, float] = {}
        self.metrics = {}

    @classmethod
    def generate_from_gcs(
        cls,
        graph_name: str,
        graph: Graph,
        save_to_file: bool = True,
        start_from_checkpoint: bool = False,
    ):
        if start_from_checkpoint:
            file_path = cls.lbg_file_path_from_name(graph_name, is_checkpoint=True)
            lbg = cls.load_from_file(file_path)
            lbg._graph = graph
            start_time = time.time()
            # start_time = time.time() - lbg.metrics["construction_time"]
            logger.info("Loaded checkpoint")
        else:
            lbg = cls(graph_name, graph.source_name, graph.target_name)
            lbg._graph = graph
            start_time = time.time()
            lbg.generate_lbg_vertices_edges_from_triplets()
            if save_to_file:
                lbg.save_to_file(lbg.lbg_file_path, is_checkpoint=True)
                duration = time.time() - start_time
                logger.info("Saved checkpoint after %.2f seconds", duration)

        lbg.generate_zero_cost_lbg_edges()

        duration = time.time() - start_time
        logger.info("Finished generating lower bound graph in %.2f seconds", duration)
        logger.info(
            f"duration in H:M:S {time.strftime('%H:%M:%S', time.gmtime(duration))}"
        )
        logger.info(f"Vertices: {len(lbg._vertices)}, Edges: {len(lbg._edges)}")

        lbg.metrics["construction_time"] = duration
        # lbg.run_dijkstra()

        if save_to_file:
            lbg.save_to_file(lbg.lbg_file_path)
        return lbg

    def generate_zero_cost_lbg_edges(self):
        logger.debug("before zero cost edges: %d", len(self._edges))
        # All lbg vertices with parent edge either u -> v or v -> u
        completed_edges = set()
        for parent_edge in self._graph.edges.values():
            if parent_edge.key in completed_edges:
                continue
            reverse_key = Edge.key_from_uv(parent_edge.v, parent_edge.u)
            group = (
                self._parent_edge_to_vertices[parent_edge.key]
                + self._parent_edge_to_vertices[reverse_key]
            )
            pairs = itertools.permutations(group, 2)
            for u_key, v_key in pairs:
                self.add_edge(u_key, v_key, 0)

        logger.debug("after zero cost edges: %d", len(self._edges))

    def generate_lbg_vertices_edges_from_triplets(self):
        graph = self._graph

        # Remove Source and Target vertices from the graph
        graph.remove_vertex(graph.source_name)
        graph.remove_vertex(graph.target_name)

        lbg = self
        logger.debug("Processing vertices of original graph...")
        # Build the set of non-zero cost lb edge triplets (pred, v, succ)
        # Assumes edges between regions are bidirectional, only want to keep one direction
        triplets = []
        for vertex in graph.vertices:
            for incoming_edge in graph.incoming_edges(vertex):
                for outgoing_edge in graph.outgoing_edges(vertex):
                    pred = incoming_edge.u
                    succ = outgoing_edge.v

                    if pred == succ or (succ, vertex, pred) in triplets:
                        continue
                    triplets.append((pred, vertex, succ))

        for pred, vertex, succ in tqdm(triplets, desc="Triplets"):
            graph.set_source(pred)
            graph.set_target(succ)
            in_edge = Edge(pred, vertex)
            out_edge = Edge(vertex, succ)
            sol = graph.solve_convex_restriction(
                active_edge_keys=[in_edge.key, out_edge.key],
                skip_post_solve=False,
            )
            if not sol.is_success:
                continue

            lbg_vertex_succ = LBGVertex(
                parent_triple=(pred, vertex, succ),
                point=graph.vertices[vertex].convex_set.vars.last_pos_from_all(
                    sol.ambient_path[1]
                ),
            )
            lbg_vertex_pred = LBGVertex(
                parent_triple=(succ, vertex, pred),
                point=graph.vertices[pred].convex_set.vars.last_pos_from_all(
                    sol.ambient_path[0]
                ),
            )
            lbg.add_vertex(lbg_vertex_pred)
            lbg.add_vertex(lbg_vertex_succ)
            lbg.add_edge(lbg_vertex_pred.key, lbg_vertex_succ.key, sol.cost)
            lbg.add_edge(lbg_vertex_succ.key, lbg_vertex_pred.key, sol.cost)

    # def add_parent_source_target(source: Vertex, target: Vertex):

    def add_vertex(self, LBG_vertex: LBGVertex):
        self._vertices[LBG_vertex.key] = LBG_vertex
        self._parent_vertex_to_vertices[LBG_vertex.parent_vertex].append(LBG_vertex)
        self._parent_edge_to_vertices[LBG_vertex.parent_edge].append(LBG_vertex.key)

    def add_edge(self, u: LBGVertexKey, v: LBGVertexKey, cost: float):
        self._edges[(u, v)] = cost

    def outgoing_edges(self, v: str) -> List[Tuple[str, str]]:
        """Get the outgoing edges of a vertex."""
        assert v in self._vertices
        return [edge_key for edge_key in self._edges.keys() if edge_key[0] == v]

    def incoming_edges(self, vertex_name: str) -> List[Tuple[str, str]]:
        """Get the incoming edges of a vertex."""
        assert vertex_name in self._vertices
        return [
            edge_key for edge_key in self._edges.values() if edge_key[1] == vertex_name
        ]

    def run_dijkstra(self) -> Tuple[float, List[str]]:
        start_time = time.time()
        self._g = {vertex: float("inf") for vertex in self._vertices.keys()}
        expanded = set()
        Q = []
        for source in self._parent_vertex_to_vertices[self._target_name]:
            self._g[source.key] = 0
            heap.heappush(Q, (0, source.key, []))
        while len(Q) > 0:
            cost, vertex_key, path = heap.heappop(Q)

            if vertex_key in expanded:
                continue
            expanded.add(vertex_key)
            for u, neighbor in self.outgoing_edges(vertex_key):
                if neighbor in expanded:
                    continue

                edge_cost = self._edges[(vertex_key, neighbor)]
                if self._g[neighbor] > self._g[vertex_key] + edge_cost:
                    self._g[neighbor] = self._g[vertex_key] + edge_cost
                    heap.heappush(Q, (self._g[neighbor], neighbor, path + [vertex_key]))
        duration = time.time() - start_time
        logger.info(f"Finished Dijkstra in {duration} seconds")
        logger.info(
            f"duration in H:M:S {time.strftime('%H:%M:%S', time.gmtime(duration))}"
        )

    def get_cost_to_go(self, gcs_vertex_name: str) -> float:
        lbg_vertex = self._parent_vertex_to_vertices[gcs_vertex_name][0]
        return self._g[lbg_vertex.key]

    def save_to_file(self, path: str, is_checkpoint=False):
        np.save(
            path,
            {
                "vertices": self._vertices,
                "edges": self._edges,
                "parent_vertex_to_vertices": self._parent_vertex_to_vertices,
                "g": self._g,
                "graph_name": self._graph_name,
                "source_name": self._source_name,
                "target_name": self._target_name,
                "metrics": self.metrics,
            },
        )

    @classmethod
    def load_from_file(cls, path: str) -> "LowerBoundGraph":
        data = np.load(path, allow_pickle=True).item()
        lbg = cls(data["graph_name"], data["source_name"], data["target_name"])
        lbg._vertices = data["vertices"]
        lbg._edges = data["edges"]
        lbg._parent_vertex_to_vertices = data["parent_vertex_to_vertices"]
        lbg._g = data["g"]
        # lbg.metrics = data["metrics"]
        return lbg

    @classmethod
    def load_from_name(cls, graph_name: str) -> "LowerBoundGraph":
        return cls.load_from_file(cls.lbg_file_path_from_name(graph_name))

    @property
    def lbg_file_path(self) -> str:
        return self.lbg_file_path_from_name(self._graph_name)

    @staticmethod
    def lbg_file_path_from_name(name: str, is_checkpoint: bool = False) -> str:
        return os.path.join(
            os.environ["PROJECT_ROOT"],
            "example_graphs",
            name + "_lbg" + ("_checkpoint" if is_checkpoint else "") + ".npy",
        )
