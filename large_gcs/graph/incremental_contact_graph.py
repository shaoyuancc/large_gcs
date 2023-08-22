import logging
from collections import defaultdict
from itertools import combinations, product
from multiprocessing import Pool
from typing import Dict, List, Tuple

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
from large_gcs.contact.contact_set_decision_variables import ContactSetDecisionVariables
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

        self._initialize_neighbor_generator()

    def _initialize_neighbor_generator(
        self,
    ):
        static_obstacles = self.obstacles
        unactuated_objects = self.objects
        actuated_robots = self.robots
        body_dict: Dict[str, RigidBody] = {
            body.name: body
            for body in static_obstacles + unactuated_objects + actuated_robots
        }
        obs_names = [body.name for body in static_obstacles]
        obj_names = [body.name for body in unactuated_objects]
        rob_names = [body.name for body in actuated_robots]
        movable = obj_names + rob_names

        logger.info(f"Generating contact sets for {len(body_dict)} bodies...")
        static_movable_pairs = list(product(obs_names, movable))
        movable_pairs = list(combinations(movable, 2))
        rigid_body_pairs = static_movable_pairs + movable_pairs

        logger.info(
            f"Generating contact pair modes for {len(rigid_body_pairs)} body pairs..."
        )

        body_pair_to_modes = {
            (body1, body2): generate_contact_pair_modes(
                body_dict[body1], body_dict[body2]
            )
            for body1, body2 in tqdm(rigid_body_pairs)
        }
        logger.info(
            f"Each body pair has on average {np.mean([len(modes) for modes in body_pair_to_modes.values()])} modes"
        )
        body_pair_to_mode_ids = {
            (body1, body2): [mode.id for mode in modes]
            for (body1, body2), modes in body_pair_to_modes.items()
        }
        mode_ids_to_mode: Dict[str, ContactPairMode] = {
            mode.id: mode for modes in body_pair_to_modes.values() for mode in modes
        }
        self._contact_pair_modes = mode_ids_to_mode

        assert self.workspace is not None, "Workspace must be set"

        workspace_pos_constraints = {
            body_name: body_dict[body_name].create_workspace_position_constraints(
                self.workspace, base_only=True
            )
            for body_name in movable
        }

        self.adj_modes = defaultdict(list)
        sets = []
        pairs_of_modes = []
        for body_pair, mode_ids in body_pair_to_mode_ids.items():
            objs = []
            robs = []
            additional_constraints = []
            for body_name in body_pair:
                if body_dict[body_name].mobility_type == MobilityType.UNACTUATED:
                    objs.append(body_dict[body_name])
                    additional_constraints.extend(workspace_pos_constraints[body_name])
                elif body_dict[body_name].mobility_type == MobilityType.ACTUATED:
                    robs.append(body_dict[body_name])
                    additional_constraints.extend(workspace_pos_constraints[body_name])

            vars = ContactSetDecisionVariables.from_objs_robs(objs, robs)
            base_polyhedra = {
                id: mode_ids_to_mode[id].create_base_polyhedron(
                    vars=vars, additional_constraints=additional_constraints
                )
                for id in mode_ids
            }
            tmp_pairs_of_modes = list(combinations(mode_ids, 2))
            pairs_of_modes.extend(tmp_pairs_of_modes)
            sets.extend(
                [
                    (base_polyhedra[mode_id1], base_polyhedra[mode_id2])
                    for mode_id1, mode_id2 in tmp_pairs_of_modes
                ]
            )
        with Pool() as pool:
            intersections = list(
                tqdm(pool.imap(self._check_intersection, sets), total=len(sets))
            )
            for (mode_id1, mode_id2), intersection in zip(
                pairs_of_modes, intersections
            ):
                if intersection:
                    self.adj_modes[mode_id1].append(mode_id2)
                    self.adj_modes[mode_id2].append(mode_id1)
