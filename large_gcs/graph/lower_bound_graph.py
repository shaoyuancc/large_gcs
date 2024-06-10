import heapq as heap
import itertools
import logging
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from multiprocessing import Pool
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydrake.all import (Intersection, ConvexSet as DrakeConvexSet)
from tqdm import tqdm

from large_gcs.graph.graph import Edge, Graph, Vertex

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

# LBGEdgeKey = namedtuple("LBGEdgeKey", ["u", "v"])
# @dataclass
# class LBGEdge:
#     u: LBGVertexKey
#     v: LBGVertexKey
#     cost: float

#     @property
#     def key(self):
#         return (self.u, self.v)



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
        self._g: Dict[LBGVertexKey, float] = {}
        self.metrics = {}
        # with Pool() as pool:
        #     inputs = [(vertex, list(
        #         itertools.product(
        #             self._graph.incoming_edges(vertex),
        #             self._graph.outgoing_edges(vertex),
        #         )
        #     )) for vertex in self._graph.vertices]
        #     vertices_to_add, edges_to_add = list(
        #         tqdm(pool.imap(self.get_lbg_vertices_edges_for_gcs_vertex, inputs), total=len(inputs))
        #     )         
        
    
    @classmethod
    def generate_from_gcs(cls, graph_name: str, graph: Graph, save_to_file: bool = True):

        lbg=cls(graph_name, graph.source_name, graph.target_name)
        lbg._graph = graph
        start_time = time.time()
        logger.debug("Processing vertices of original graph...")
        for vertex in tqdm(graph.vertices, desc="Vertices"):
            logger.debug("Findings solutions for triplets...")
            edge_pairs = list(
                itertools.product(
                    graph.incoming_edges(vertex),
                    graph.outgoing_edges(vertex),
                )
            )
            for incoming_edge, outgoing_edge in tqdm(edge_pairs, desc="Edge pairs"):
                predecessor = incoming_edge.u
                successor = outgoing_edge.v
                if predecessor == successor:
                    continue

                graph.set_source(predecessor)
                graph.set_target(successor)
                sol = graph.solve_convex_restriction(
                    active_edge_keys=[incoming_edge.key, outgoing_edge.key],
                    skip_post_solve=False,
                )
                if not sol.is_success:
                    continue

                lbg_vertex_pred = LBGVertex(
                    parent_triple=(predecessor, vertex, successor),
                    parent_vertex=predecessor,
                    point=graph.vertices[predecessor].convex_set.vars.last_pos_from_all(sol.ambient_path[0]),
                )
                lbg_vertex_succ = LBGVertex(
                    parent_triple=(predecessor, vertex, successor),
                    parent_vertex=successor,
                    point=graph.vertices[vertex].convex_set.vars.last_pos_from_all(sol.ambient_path[1]),
                )
                lbg.add_vertex(lbg_vertex_pred)
                lbg.add_vertex(lbg_vertex_succ)
                lbg.add_edge(lbg_vertex_pred.key, lbg_vertex_succ.key, sol.cost)
                lbg.add_edge(lbg_vertex_succ.key, lbg_vertex_pred.key, sol.cost)
        logger.debug("Generating 0 cost edges within intersections... (Parallel)")
        with Pool() as pool:
            inputs = []
            for edge in graph.edges:
                u = graph.edges[edge].u
                v = graph.edges[edge].v
                u_set = graph.vertices[u].convex_set.base_set
                v_set = graph.vertices[v].convex_set.base_set
                inputs.append((lbg._parent_vertex_to_vertices[u] + lbg._parent_vertex_to_vertices[v],
                    u_set, v_set))
            lbg_edges = list(tqdm(pool.imap(lbg.get_lbg_edges_in_intersection, inputs), total=len(inputs)))
            for edges in lbg_edges:
                for u_key, v_key in edges:
                    lbg.add_edge(u_key, v_key, 0)
                    lbg.add_edge(v_key, u_key, 0)
        duration = time.time() - start_time
        logger.info("Finished generating lower bound graph in %.2f seconds", duration)
        logger.info(
            f"duration in H:M:S {time.strftime('%H:%M:%S', time.gmtime(duration))}"
        )
        logger.info(f"Vertices: {len(lbg._vertices)}, Edges: {len(lbg._edges)}")

        lbg.metrics["construction_time"] = duration
        lbg.run_dijkstra()
        
        if save_to_file:
            lbg.save_to_file(lbg.lbg_file_path)
        
    
    @staticmethod
    def get_lbg_vertices_edges_for_gcs_vertex(vertex: Vertex, edges: List[Edge]):
        return True

    @staticmethod
    def get_lbg_edges_in_intersection(args) -> List[Tuple[LBGVertexKey, LBGVertexKey]]:
        potential_lbg_vertices, u_set, v_set = args
        intersection = Intersection(u_set, v_set)
        points_in_intersection = [v.key for v in potential_lbg_vertices if intersection.PointInSet(v.point)]
        return list(itertools.permutations(points_in_intersection, 2))

    def add_vertex(self, LBG_vertex: LBGVertex):
        self._vertices[LBG_vertex.key] = LBG_vertex
        self._parent_vertex_to_vertices[LBG_vertex.parent_vertex].append(LBG_vertex)
        
    def add_edge(self, u: LBGVertexKey, v: LBGVertexKey, cost: float):
        self._edges[(u, v)] = cost

    def outgoing_edges(self, v: str) -> List[Tuple[str, str]]:
        """Get the outgoing edges of a vertex."""
        assert v in self._vertices
        return [edge_key for edge_key in self._edges.keys() if edge_key[0] == v]

    def incoming_edges(self, vertex_name: str) -> List[Tuple[str, str]]:
        """Get the incoming edges of a vertex."""
        assert vertex_name in self._vertices
        return [edge_key for edge_key in self._edges.values() if edge_key[1] == vertex_name]

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
            for (u, neighbor) in self.outgoing_edges(vertex_key):
                if neighbor in expanded:
                    continue

                edge_cost = self._edges[(vertex_key, neighbor)]
                if self._g[neighbor] > self._g[vertex_key] + edge_cost:
                    self._g[neighbor] = self._g[vertex_key] + edge_cost
                    heap.heappush(
                        Q, (self._g[neighbor], neighbor, path + [vertex_key])
                    )
        duration = time.time() - start_time
        logger.info(f"Finished Dijkstra in {duration} seconds")
        logger.info(
            f"duration in H:M:S {time.strftime('%H:%M:%S', time.gmtime(duration))}"
        )

    def get_cost_to_go(self, gcs_vertex_name: str) -> float:
        lbg_vertex = self._parent_vertex_to_vertices[gcs_vertex_name][0]
        return self._g[lbg_vertex.key]

    def save_to_file(self, path: str):
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
            },
        )
    

    @classmethod
    def load_from_file(
        cls,
        path: str
    ) -> "LowerBoundGraph":
        data = np.load(path, allow_pickle=True).item()
        lbg = cls(data["graph_name"], data["source_name"], data["target_name"])
        lbg._vertices = data["vertices"]
        lbg._edges = data["edges"]
        lbg._parent_vertex_to_vertices = data["parent_vertex_to_vertices"]
        lbg._g = data["g"]
        return lbg
    
    @classmethod
    def load_from_name(
        cls,
        graph_name: str
    ) -> "LowerBoundGraph":
        return cls.load_from_file(cls.lbg_file_path_from_name(graph_name))
    
    @property
    def lbg_file_path(self) -> str:
        return self.lbg_file_path_from_name(self._graph_name)
    
    @staticmethod
    def lbg_file_path_from_name(name: str) -> str:
        return os.path.join(os.environ["PROJECT_ROOT"], "example_graphs", name + "_lbg.npy")

