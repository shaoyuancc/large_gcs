import logging
from itertools import combinations, product
from typing import List, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import Constraint, Cost, Expression, GraphOfConvexSets, eq
from tqdm import tqdm

from large_gcs.contact.contact_pair_mode import (
    ContactPairMode,
    InContactPairMode,
    generate_contact_pair_modes,
)
from large_gcs.contact.contact_regions_set import ContactRegionParams, ContactRegionsSet
from large_gcs.contact.contact_set import ContactPointSet, ContactSet
from large_gcs.contact.rigid_body import BodyColor, MobilityType, RigidBody
from large_gcs.geometry.polyhedron import Polyhedron
from large_gcs.graph.contact_cost_constraint_factory import (
    edge_constraint_position_continuity,
    edge_cost_constant,
    vertex_cost_force_actuation_norm,
    vertex_cost_position_path_length,
)
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.graph import Graph, ShortestPathSolution

logger = logging.getLogger(__name__)


class IncrementalContactGraph(ContactGraph):
    def __init__(
        self,
        static_obstacles: List[RigidBody],
        unactuated_objects: List[RigidBody],
        actuated_robots: List[RigidBody],
        source_pos_objs: List[np.ndarray],
        source_pos_robs: List[np.ndarray],
        target_pos_objs: List[np.ndarray] = None,
        target_pos_robs: List[np.ndarray] = None,
        target_region_params: List[ContactRegionParams] = None,
        workspace: np.ndarray = None,
        vertex_exclusion: List[str] = None,
        vertex_inclusion: List[str] = None,
        contact_pair_modes: List[ContactPairMode] = None,  # For loading a saved graph
        contact_set_mode_ids: List[List[str]] = None,  # For loading a saved graph
        edge_keys: List[Tuple[str, str]] = None,  # For loading a saved graph
    ):
        """
        Args:
            static_obstacles: List of static obstacles.
            unactuated_objects: List of unactuated objects.
            actuated_robots: List of actuated robots.
            initial_positions: List of initial positions of.
        """
        Graph.__init__(self, workspace=workspace)
        # Note: The order of operations in this constructor is important
        self.vertex_inclusion = vertex_inclusion
        self.vertex_exclusion = vertex_exclusion
        self.obstacles = None
        self.objects = None
        self.robots = None
        self.target_pos = None
        self.target_region_params = None
        self.target_regions = None

        for thing in static_obstacles:
            assert (
                thing.mobility_type == MobilityType.STATIC
            ), f"{thing.name} is not static"
        for thing in unactuated_objects:
            assert (
                thing.mobility_type == MobilityType.UNACTUATED
            ), f"{thing.name} is not unactuated"
        for thing in actuated_robots:
            assert (
                thing.mobility_type == MobilityType.ACTUATED
            ), f"{thing.name} is not actuated"
        self.obstacles = static_obstacles
        self.objects = unactuated_objects
        self.robots = actuated_robots

        sets = []
        set_ids = []
        self.source_pos = source_pos_objs + source_pos_robs

        sets += [
            ContactPointSet(
                "source", self.objects, self.robots, source_pos_objs, source_pos_robs
            )
        ]

        if target_pos_objs is not None and target_pos_robs is not None:
            self.target_pos = target_pos_objs + target_pos_robs
            sets += [
                ContactPointSet(
                    "target",
                    self.objects,
                    self.robots,
                    target_pos_objs,
                    target_pos_robs,
                ),
            ]
        elif target_region_params is not None:

            self.target_region_params = target_region_params
            self.target_regions = [
                Polyhedron.from_vertices(params.region_vertices)
                for params in target_region_params
            ]
            sets.append(
                ContactRegionsSet(
                    self.objects, self.robots, target_region_params, "target"
                )
            )
        else:
            raise ValueError("Must specify either target_pos or target_regions")

        set_ids += ["source", "target"]

        # Add convex sets to graph (Need to do this before generating edges)
        self.add_vertices_from_sets(
            sets,
            costs=self._create_vertex_costs(sets),
            constraints=self._create_vertex_constraints(sets),
            names=set_ids,
        )
        self.set_source("source")
        self.set_target("target")
