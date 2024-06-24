import heapq as heap
import itertools
import logging
import os
import time
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from large_gcs.graph.contact_graph import ContactGraph
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
        self._adjacency_list: Dict[str, List[str]] = defaultdict(list)
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
                lbg.save_to_file(
                    lbg.lbg_file_path_from_name(graph_name, is_checkpoint=True)
                )
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

        if save_to_file:
            lbg.save_to_file(lbg.lbg_file_path)
        return lbg

    def generate_lbg_vertices_edges_from_triplets(self):
        # Remove Source and Target vertices from the graph
        self._graph.remove_vertex(self._graph.source_name)
        self._graph.remove_vertex(self._graph.target_name)

        triplets, paths = self.enumerate_triplets()
        self.process_triplets_parallel(triplets, paths)
        # self.process_triplets(triplets, paths)

    def enumerate_triplets(self):
        # Build the set of non-zero cost lb edge triplets (pred, v, succ)
        # Assumes edges between regions are bidirectional, only want to keep one direction
        logger.debug(f"Enumerating Triplets for {self._graph.n_vertices} vertices...")
        triplets = []
        paths = []
        for vertex in tqdm(self._graph.vertices, desc="Vertices"):
            out_values = self._graph.successors(vertex)
            for i in range(len(out_values)):
                for j in range(i + 1, len(out_values)):
                    active_edges = [
                        Edge.key_from_uv(out_values[i], vertex),
                        Edge.key_from_uv(vertex, out_values[j]),
                    ]
                    triplets.append((out_values[i], vertex, out_values[j]))
                    paths.append(active_edges)
        return triplets, paths

    def process_triplets_parallel(self, triplets, paths):
        logger.debug(f"Processing {len(triplets)} Triplets in parallel...")
        batch_size = 4096
        total = len(triplets)
        with tqdm(total=total, desc="Processing batches", unit="batch") as pbar:
            for batch_start in range(0, total, batch_size):
                batch_end = min(batch_start + batch_size, total)
                batch_triplets = triplets[batch_start:batch_end]
                batch_paths = paths[batch_start:batch_end]
                batch_sols = self._graph.solve_convex_restrictions(batch_paths)
                # import pdb; pdb.set_trace()
                for (pred, vertex, succ), sol in zip(batch_triplets, batch_sols):
                    # logger.debug(sol)
                    if not sol.is_success:
                        continue

                    lbg_vertex_succ = LBGVertex(
                        parent_triple=(pred, vertex, succ),
                        point=self._graph.vertices[
                            vertex
                        ].convex_set.vars.last_pos_from_all(sol.ambient_path[1]),
                    )
                    lbg_vertex_pred = LBGVertex(
                        parent_triple=(succ, vertex, pred),
                        point=self._graph.vertices[
                            pred
                        ].convex_set.vars.last_pos_from_all(sol.ambient_path[0]),
                    )
                    self.add_vertex(lbg_vertex_pred)
                    self.add_vertex(lbg_vertex_succ)
                    self.add_edge(lbg_vertex_pred.key, lbg_vertex_succ.key, sol.cost)
                    self.add_edge(lbg_vertex_succ.key, lbg_vertex_pred.key, sol.cost)
                pbar.update(batch_end - batch_start)

    def process_triplets(self, triplets, paths):
        logger.debug(f"Processing {len(triplets)} Triplets...")
        for (pred, vertex, succ), active_edges in tqdm(
            zip(triplets, paths), total=len(triplets), desc="Triplets"
        ):
            self._graph.set_source(pred)
            self._graph.set_target(succ)

            sol = self._graph.solve_convex_restriction(
                active_edge_keys=active_edges,
                skip_post_solve=False,
            )
            if not sol.is_success:
                continue

            lbg_vertex_succ = LBGVertex(
                parent_triple=(pred, vertex, succ),
                point=self._graph.vertices[vertex].convex_set.vars.last_pos_from_all(
                    sol.ambient_path[1]
                ),
            )
            lbg_vertex_pred = LBGVertex(
                parent_triple=(succ, vertex, pred),
                point=self._graph.vertices[pred].convex_set.vars.last_pos_from_all(
                    sol.ambient_path[0]
                ),
            )
            self.add_vertex(lbg_vertex_pred)
            self.add_vertex(lbg_vertex_succ)
            self.add_edge(lbg_vertex_pred.key, lbg_vertex_succ.key, sol.cost)
            self.add_edge(lbg_vertex_succ.key, lbg_vertex_pred.key, sol.cost)

    def generate_zero_cost_lbg_edges(self):
        count = 0
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
            completed_edges.add(parent_edge.key)
            completed_edges.add(reverse_key)
            pairs = itertools.pairwise(group)
            for u_key, v_key in pairs:
                self.add_edge(u_key, v_key, 0)
                self.add_edge(v_key, u_key, 0)
                count += 2

        logger.debug(f"Added {count} zero cost edges to LBG")

    def update_lbg(self, parent_vertex_name: str, parent_vertex: Vertex):
        """Assumes that there's no source and target already in the gcs
        graph."""
        logger.debug(f"Checking intersections with parent vertex")
        v_names = self._graph.vertex_names
        sets = [
            (parent_vertex.convex_set.base_set, v.convex_set.base_set)
            for v in self._graph.vertices.values()
        ]
        batch_size = 100
        total = len(sets)
        lbg_vertex = LBGVertex(
            ("", "", parent_vertex_name), parent_vertex.convex_set.center
        )
        self.add_vertex(lbg_vertex)
        with tqdm(total=total, desc="Processing batches", unit="batch") as pbar:
            for batch_start in range(0, total, batch_size):
                batch_end = min(batch_start + batch_size, total)
                batch_v_names = v_names[batch_start:batch_end]
                batch_sets = sets[batch_start:batch_end]
                batch_intersections = []
                with Pool() as pool:
                    batch_intersections += list(
                        pool.imap(ContactGraph._check_intersection, batch_sets)
                    )
                for v_name, intersection in zip(batch_v_names, batch_intersections):
                    if intersection:
                        u = self._parent_vertex_to_vertices[v_name][0]
                        self.add_edge(u, lbg_vertex.key, 0)
                        self.add_edge(lbg_vertex.key, u, 0)
                pbar.update(batch_end - batch_start)

    def add_vertex(self, LBG_vertex: LBGVertex):
        self._vertices[LBG_vertex.key] = LBG_vertex
        self._parent_vertex_to_vertices[LBG_vertex.parent_vertex].append(LBG_vertex.key)
        self._parent_edge_to_vertices[LBG_vertex.parent_edge].append(LBG_vertex.key)

    def add_edge(self, u: LBGVertexKey, v: LBGVertexKey, cost: float):
        self._edges[(u, v)] = cost
        self._adjacency_list[u].append(v)

    def outgoing_edges(self, v: str) -> List[Tuple[str, str]]:
        """Get the outgoing edges of a vertex."""
        # assert v in self._vertices
        return [edge_key for edge_key in self._edges.keys() if edge_key[0] == v]
        # return self._adjacency_list[v]

    def successors(self, v: LBGVertexKey) -> List[LBGVertexKey]:
        return self._adjacency_list[v]

    def incoming_edges(self, vertex_name: str) -> List[Tuple[str, str]]:
        """Get the incoming edges of a vertex."""
        assert vertex_name in self._vertices
        return [
            edge_key for edge_key in self._edges.values() if edge_key[1] == vertex_name
        ]

    def run_dijkstra(self, parent_vertex_start) -> Tuple[float, List[str]]:
        start_time = time.time()
        self._g = {vertex: float("inf") for vertex in self._vertices.keys()}
        expanded = set()
        Q = []
        for source in self._parent_vertex_to_vertices[parent_vertex_start]:
            self._g[source] = 0
            heap.heappush(Q, (0, source, []))
        while len(Q) > 0:
            cost, vertex_key, path = heap.heappop(Q)
            if vertex_key in expanded:
                continue
            expanded.add(vertex_key)
            for neighbor in self.successors(vertex_key):
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
        return self._g[lbg_vertex]

    def save_to_file(self, path: str):
        np.save(
            path,
            {
                "vertices": self._vertices,
                "edges": self._edges,
                "adjacency_list": self._adjacency_list,
                "parent_vertex_to_vertices": self._parent_vertex_to_vertices,
                "parent_edge_to_vertices": self._parent_edge_to_vertices,
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
        lbg._adjacency_list = data["adjacency_list"]
        lbg._parent_edge_to_vertices = data["parent_edge_to_vertices"]
        lbg._parent_vertex_to_vertices = data["parent_vertex_to_vertices"]
        lbg._g = data["g"]
        lbg.metrics = data["metrics"]
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
