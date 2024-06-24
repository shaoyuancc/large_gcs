import logging
import os
import pickle
from typing import List

import numpy as np
import scipy
from pydrake.all import HPolyhedron

from large_gcs.geometry.point import Point
from large_gcs.geometry.polyhedron import Polyhedron
from large_gcs.graph.cfree_cost_constraint_factory import (
    create_cfree_continuity_edge_constraint,
    create_cfree_l2norm_vertex_cost,
    create_region_target_edge_constraint,
    create_source_region_edge_constraint,
)
from large_gcs.graph.graph import Edge, Graph, Vertex

logger = logging.getLogger(__name__)


class CFreeGraph(Graph):
    def __init__(
        self,
        source: np.ndarray,
        target: np.ndarray,
        source_poly_idx: int,
        target_poly_idx: int,
        hpolys: List[HPolyhedron],
        adj_list: List,
    ) -> None:
        """Assumes regions are polyhedra, source and target are points.

        Source is within region 0, target is within region 1.
        """
        super().__init__()
        base_dim = hpolys[0].A().shape[1]

        # Create source and target vertices
        source_set = Vertex(convex_set=Point(source))
        self.add_vertex(vertex=source_set, name="source")
        target_set = Vertex(convex_set=Point(target))
        self.add_vertex(vertex=target_set, name="target")

        # Add vertices for each region
        for i, hpoly in enumerate(hpolys):
            # Create set with 2 knot points
            A = scipy.linalg.block_diag(hpoly.A(), hpoly.A())
            b = np.hstack((hpoly.b(), hpoly.b()))
            poly = Polyhedron(A, b, should_compute_vertices=False)
            poly.create_nullspace_set()
            v = Vertex(
                convex_set=poly, costs=[create_cfree_l2norm_vertex_cost(base_dim)]
            )
            self.add_vertex(vertex=v, name=f"region_{i}")

        # logger.debug(f"Adjacency list: {pprint.pformat(adj_list)}")
        for i, j, offset in adj_list:
            e = Edge(
                f"region_{i}",
                f"region_{j}",
                constraints=[create_cfree_continuity_edge_constraint(offset)],
            )
            self.add_edge(edge=e)

        # Add edges from source and to target
        source_edge = Edge(
            "source",
            f"region_{source_poly_idx}",
            constraints=[create_source_region_edge_constraint(base_dim)],
        )
        self.add_edge(edge=source_edge)
        target_edge = Edge(
            f"region_{target_poly_idx}",
            "target",
            constraints=[create_region_target_edge_constraint(base_dim)],
        )
        self.add_edge(edge=target_edge)

        self.set_source("source")
        self.set_target("target")

    @classmethod
    def load_from_file(
        cls,
        graph_name: str,
        source: np.ndarray,
        target: np.ndarray,
        source_poly_idx: int,
        target_poly_idx: int,
    ) -> "CFreeGraph":
        base = os.path.join(
            os.environ["PROJECT_ROOT"], "example_graphs", f"{graph_name}"
        )
        regions_file = f"{base}_regions.pkl"
        adj_file = f"{base}_adj.pkl"
        with open(regions_file, "rb") as f:
            regions = pickle.load(f)
        with open(adj_file, "rb") as f:
            adj_list = pickle.load(f)

        return cls(
            source=source,
            target=target,
            source_poly_idx=source_poly_idx,
            target_poly_idx=target_poly_idx,
            hpolys=regions,
            adj_list=adj_list,
        )

    @staticmethod
    def graph_file_path_from_name(name: str) -> str:
        return
