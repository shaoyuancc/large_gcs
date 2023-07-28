import logging
from itertools import combinations, product
from typing import List, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import Constraint, Cost, Expression, GraphOfConvexSets, eq
from tqdm import tqdm

from large_gcs.contact.contact_pair_mode import generate_contact_pair_modes
from large_gcs.contact.contact_set import ContactPointSet, ContactSet
from large_gcs.contact.rigid_body import BodyColor, MobilityType, RigidBody
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.graph import Graph, ShortestPathSolution

logger = logging.getLogger(__name__)


class FactoredCollisionFreeGraph(ContactGraph):
    def __init__(
        self,
        movable_body: RigidBody,
        static_obstacles: List[RigidBody],
        target_pos: np.ndarray,
        workspace: np.ndarray = None,
    ):
        self.vertices = {}
        self.edges = {}
        self._source_name = None
        self._target_name = None
        self._default_costs_constraints = None
        self.movable_body = movable_body
        self.workspace = workspace
        self.obstacles = []
        self.objects = []
        self.robots = []
        self._gcs = GraphOfConvexSets()

        for thing in static_obstacles:
            assert (
                thing.mobility_type == MobilityType.STATIC
            ), f"{thing.name} is not static"
        if movable_body.mobility_type == MobilityType.UNACTUATED:
            self.objects = [movable_body]
            target_pos_objs = [target_pos]
            target_pos_robs = []
        elif movable_body.mobility_type == MobilityType.ACTUATED:
            self.robots = [movable_body]
            target_pos_robs = [target_pos]
            target_pos_objs = []
        else:
            raise ValueError(
                f"Mobility type for movable body {movable_body.mobility_type} not supported"
            )
        self.source_pos = None
        self.target_pos = [target_pos]
        self.obstacles = static_obstacles

        sets, set_ids = self._generate_contact_sets()
        sets.append(
            ContactPointSet(
                "target", self.objects, self.robots, target_pos_objs, target_pos_robs
            )
        )
        set_ids.append("target")

        # Add convex sets to graph (Need to do this before generating edges)
        self.add_vertices_from_sets(
            sets,
            costs=self._create_vertex_costs(sets),
            constraints=self._create_vertex_constraints(sets),
            names=set_ids,
        )
        self.set_target("target")
        edges = self._generate_contact_graph_edges(set_ids)
        self.add_edges_from_vertex_names(
            *zip(*edges),
            costs=self._create_edge_costs(edges),
            constraints=self._create_edge_constraints(edges),
        )

        # Check that the target is reachable
        if len(self.incoming_edges(self.target_name)) == 0:
            logger.warn("Target is not reachable from any other set")

        logger.info(
            f"Created factored collision free graph for {movable_body.name}: {self.params}"
        )

    def _generate_contact_sets(self) -> Tuple[List[ContactSet], List[str]]:
        body_dict = {
            body.name: body for body in self.obstacles + self.objects + self.robots
        }
        obs_names = [obs.name for obs in self.obstacles]

        rigid_body_pairs = list(product(obs_names, [self.movable_body.name]))
        body_pair_to_modes = {
            (body1, body2): generate_contact_pair_modes(
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

        additional_constraints = []
        if self.workspace is not None:
            additional_constraints = (
                self.movable_body.create_workspace_position_constraints(self.workspace)
            )

        all_contact_sets = [
            ContactSet.from_factored_collision_free_body(
                [mode_ids_to_mode[mode_id] for mode_id in set_id],
                self.movable_body,
                additional_constraints=additional_constraints,
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
