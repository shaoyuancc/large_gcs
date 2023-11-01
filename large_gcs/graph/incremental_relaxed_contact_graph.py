import logging
from itertools import combinations, product
from typing import Dict, Iterable

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import Constraint, Cost, Expression, GraphOfConvexSets, eq
from tqdm import tqdm

from large_gcs.contact.contact_pair_mode import (
    ContactPairMode,
    InContactPairMode,
    RelaxedInContactPairMode,
    generate_contact_pair_modes,
    generate_relaxed_contact_pair_modes,
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
from large_gcs.graph.graph import Edge, Graph, ShortestPathSolution, Vertex
from large_gcs.graph.incremental_contact_graph import IncrementalContactGraph

logger = logging.getLogger(__name__)


class IncrementalRelaxedContactGraph(IncrementalContactGraph):
    @classmethod
    def from_inc_contact_graph(self, inc_contact_graph: IncrementalContactGraph):
        if inc_contact_graph.target_pos is None:
            target_pos_objs = None
            target_pos_robs = None
        else:
            target_pos_objs = inc_contact_graph.target_pos[
                : len(inc_contact_graph.objects)
            ]
            target_pos_robs = inc_contact_graph.target_pos[
                len(inc_contact_graph.objects) :
            ]
        super().__init__(
            static_obstacles=inc_contact_graph.obstacles,
            unactuated_objects=inc_contact_graph.objects,
            actuated_robots=inc_contact_graph.robots,
            source_pos_objs=inc_contact_graph.source_pos[
                : len(inc_contact_graph.objects)
            ],
            source_pos_robs=inc_contact_graph.source_pos[
                len(inc_contact_graph.objects) :
            ],
            target_pos_objs=target_pos_objs,
            target_pos_robs=target_pos_robs,
            target_region_params=inc_contact_graph.target_region_params,
            workspace=inc_contact_graph.workspace,
            should_incl_simul_mode_switches=inc_contact_graph._should_incl_simul_mode_switches,
            should_add_gcs=inc_contact_graph._should_add_gcs,
            should_add_const_edge_cost=inc_contact_graph._should_add_const_edge_cost,
        )

    def _create_contact_set_from_contact_pair_mode_ids(
        self, mode_ids: Iterable[str]
    ) -> ContactSet:
        # No additional force constraints because for RIC, it is all added in the contact pair mode
        # No other force constraints in other modes for the RIC graph.

        # Objects not in some RIC must have 0 resultant force.
        no_contact_objs = set([obj.name for obj in self.objects])
        for mode_id in mode_ids:
            mode = self._contact_pair_modes[mode_id]
            if isinstance(mode, RelaxedInContactPairMode):
                no_contact_objs.discard(mode.body_a.name)
                no_contact_objs.discard(mode.body_b.name)
        set_force_constraints = []
        for obj_name in no_contact_objs:
            set_force_constraints += eq(
                self._body_dict[obj_name].vars_force_res, 0
            ).tolist()

        for body_name in self._movable:
            body = self._body_dict[body_name]
            set_force_constraints += body.force_constraints

        contact_set = ContactSet.from_objs_robs(
            [self._contact_pair_modes[mode_id] for mode_id in mode_ids],
            self.objects,
            self.robots,
            additional_constraints=set_force_constraints + self._workspace_constraints,
            additional_base_constraints=self._base_workspace_constraints,
        )

        return contact_set

    def _initialize_set_generation_variables(self):
        self._body_dict: Dict[str, RigidBody] = {
            body.name: body for body in self.obstacles + self.objects + self.robots
        }

        obs_names = [body.name for body in self.obstacles]
        obj_names = [body.name for body in self.objects]
        rob_names = [body.name for body in self.robots]
        movable = obj_names + rob_names
        self._movable = movable

        self._workspace_constraints = []
        self._base_workspace_constraints = []
        for body in self.objects + self.robots:
            body.create_workspace_position_constraints(self.workspace)
            self._workspace_constraints += body.workspace_constraints
            self._base_workspace_constraints += body.base_workspace_constraints

        static_movable_pairs = list(product(obs_names, movable))
        obj_obj_pairs = list(combinations(obj_names, 2))
        obj_rob_pairs = list(product(obj_names, rob_names))
        movable_pairs = obj_obj_pairs + obj_rob_pairs
        # The thing that's excluded is rob_rob pairs, we don't consider them in the relaxed contact graph.

        logger.info(
            f"Generating contact pair modes for {len(static_movable_pairs) + len(movable_pairs)} body pairs..."
        )

        body_pair_to_modes = {}
        for body1, body2 in tqdm(static_movable_pairs):
            body_pair_to_modes[(body1, body2)] = generate_contact_pair_modes(
                self._body_dict[body1], self._body_dict[body2]
            )
        for body1, body2 in tqdm(movable_pairs):
            body_pair_to_modes[(body1, body2)] = generate_relaxed_contact_pair_modes(
                self._body_dict[body1], self._body_dict[body2]
            )

        self._body_pair_to_mode_ids = {
            (body1, body2): [mode.id for mode in modes]
            for (body1, body2), modes in body_pair_to_modes.items()
        }
        mode_ids_to_mode: Dict[str, ContactPairMode] = {
            mode.id: mode for modes in body_pair_to_modes.values() for mode in modes
        }
        self._contact_pair_modes = mode_ids_to_mode
