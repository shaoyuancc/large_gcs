import logging
from itertools import combinations, product
from typing import List, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import Constraint, Cost, Expression, GraphOfConvexSets, eq
from tqdm import tqdm

from large_gcs.contact.contact_pair_mode import generate_cfree_contact_pair_modes
from large_gcs.contact.contact_regions_set import ContactRegionParams, ContactRegionsSet
from large_gcs.contact.contact_set import ContactPointSet, ContactSet
from large_gcs.contact.rigid_body import BodyColor, MobilityType, RigidBody
from large_gcs.geometry.polyhedron import Polyhedron
from large_gcs.graph.contact_cost_constraint_factory import (
    vertex_cost_position_path_length,
)
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.graph import Graph, ShortestPathSolution

logger = logging.getLogger(__name__)


class FactoredCollisionFreeGraph(ContactGraph):
    def __init__(
        self,
        movable_body: RigidBody,
        static_obstacles: List[RigidBody],
        source_pos: np.ndarray,
        target_pos: np.ndarray = None,
        target_region_params: ContactRegionParams = None,
        cost_scaling: float = 1.0,
        workspace: np.ndarray = None,
        add_source_set: bool = False,
    ):
        Graph.__init__(self, workspace=workspace)
        self._cost_scaling = cost_scaling
        self.movable_body = movable_body
        self.obstacles = []
        self.objects = []
        self.robots = []
        self.target_region_params = None
        self.target_regions = None
        rob_indicies = None
        obj_indices = None

        for thing in static_obstacles:
            assert (
                thing.mobility_type == MobilityType.STATIC
            ), f"{thing.name} is not static"
        if movable_body.mobility_type == MobilityType.UNACTUATED:
            self.objects = [movable_body]
            source_pos_objs = [source_pos]
            source_pos_robs = []
            target_pos_objs = [target_pos]
            target_pos_robs = []
            obj_indices = [0]
        elif movable_body.mobility_type == MobilityType.ACTUATED:
            self.robots = [movable_body]
            source_pos_robs = [source_pos]
            source_pos_objs = []
            target_pos_robs = [target_pos]
            target_pos_objs = []
            rob_indicies = [0]
        else:
            raise ValueError(
                f"Mobility type for movable body {movable_body.mobility_type} not supported"
            )
        self.source_pos = source_pos_objs + source_pos_robs

        self.target_pos = None
        self.obstacles = static_obstacles
        target_name = f"target_{self.movable_body.name}"
        sets, set_ids = self._generate_contact_sets()
        if target_pos is not None:
            self.target_pos = [target_pos]
            sets.append(
                ContactPointSet(
                    target_name,
                    self.objects,
                    self.robots,
                    target_pos_objs,
                    target_pos_robs,
                )
            )
        elif target_region_params is not None:
            self.target_region_params = [target_region_params]
            self.target_regions = [
                Polyhedron.from_vertices(target_region_params.region_vertices)
            ]
            # We override the obj and rob indices here because we are only passing a single rigid body.
            factored_params = ContactRegionParams(
                target_region_params.region_vertices,
                obj_indices=obj_indices,
                rob_indices=rob_indicies,
            )
            sets.append(
                ContactRegionsSet(
                    self.objects, self.robots, [factored_params], target_name
                )
            )
        else:
            raise ValueError("Must specify either target_pos or target_region_params")
        set_ids.append(target_name)
        if add_source_set:
            source_name = f"source_{self.movable_body.name}"
            sets += [
                ContactPointSet(
                    source_name,
                    self.objects,
                    self.robots,
                    source_pos_objs,
                    source_pos_robs,
                )
            ]
            set_ids.append(source_name)

        # Add convex sets to graph (Need to do this before generating edges)
        self.add_vertices_from_sets(
            sets,
            costs=self._create_vertex_costs(sets),
            constraints=self._create_vertex_constraints(sets),
            names=set_ids,
        )
        self.set_target(target_name)
        if add_source_set:
            self.set_source(source_name)
        edges = self._generate_contact_graph_edges(set_ids)
        self.add_edges_from_vertex_names(
            *zip(*edges),
            costs=self._create_edge_costs(edges),
            constraints=self._create_edge_constraints(edges),
        )

        # Check that the target is reachable
        if len(self.incoming_edges(self.target_name)) == 0:
            logger.warning("Target is not reachable from any other set")

        logger.info(
            f"Created factored collision free graph for {movable_body.name}: {self.params}"
        )

    def _create_vertex_costs(self, sets: List[ContactSet]) -> List[List[Cost]]:
        logger.info("Creating vertex costs for factored_collision_free_graph...")
        costs = [
            [
                vertex_cost_position_path_length(set.vars, self._cost_scaling),
            ]
            if isinstance(set, ContactSet)
            else []
            for set in tqdm(sets)
        ]
        return costs

    def _generate_contact_sets(self) -> Tuple[List[ContactSet], List[str]]:
        body_dict = {
            body.name: body for body in self.obstacles + self.objects + self.robots
        }
        obs_names = [obs.name for obs in self.obstacles]

        rigid_body_pairs = list(product(obs_names, [self.movable_body.name]))
        body_pair_to_modes = {
            (body1, body2): generate_cfree_contact_pair_modes(
                body_dict[body1], body_dict[body2]
            )
            for body1, body2 in rigid_body_pairs
        }
        body_pair_to_mode_names = {
            (body1, body2): [mode.id for mode in modes]
            for (body1, body2), modes in body_pair_to_modes.items()
        }
        mode_ids_to_mode = {
            mode.id: mode for modes in body_pair_to_modes.values() for mode in modes
        }
        # Each set is the cartesian product of the modes for each object pair
        set_ids = list(product(*body_pair_to_mode_names.values()))

        self._base_workspace_constraints = []
        for body in self.objects + self.robots:
            body.create_workspace_position_constraints(self.workspace)
            self._base_workspace_constraints += body.base_workspace_constraints

        all_contact_sets = [
            ContactSet.from_factored_collision_free_body(
                [mode_ids_to_mode[mode_id] for mode_id in set_id],
                self.movable_body,
                additional_constraints=self._base_workspace_constraints,
                additional_base_constraints=self._base_workspace_constraints,
            )
            for set_id in set_ids
        ]
        sets_to_keep = [
            contact_set
            for contact_set in all_contact_sets
            if not contact_set.set.IsEmpty()
        ]
        sets_to_keep_ids = [str(contact_set.id) for contact_set in sets_to_keep]
        return sets_to_keep, sets_to_keep_ids
