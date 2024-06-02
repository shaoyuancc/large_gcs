import ast
import logging
import re
from collections import defaultdict
from itertools import combinations, product
from typing import Dict, Iterable

import numpy as np
from pydrake.all import eq
from tqdm import tqdm

from large_gcs.contact.contact_pair_mode import (
    ContactPairMode,
    RelaxedInContactPairMode,
    generate_contact_pair_modes,
    generate_relaxed_contact_pair_modes,
)
from large_gcs.contact.contact_set import ContactSet
from large_gcs.contact.rigid_body import MobilityType, RigidBody
from large_gcs.graph.incremental_contact_graph import IncrementalContactGraph

logger = logging.getLogger(__name__)


class IncrementalRelaxedContactGraph(IncrementalContactGraph):
    @classmethod
    def from_inc_contact_graph(cls, inc_contact_graph: IncrementalContactGraph):
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
        return cls(
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
        # Calculate contact chains with no robots
        # Objects not in some RIC must have 0 velocity.
        no_contact_objs = self._get_no_rob_contact_objs(mode_ids)
        set_force_constraints = []
        for obj_name in no_contact_objs:
            vars_vel = self._body_dict[obj_name].vars_vel
            set_force_constraints += eq(vars_vel, np.zeros_like(vars_vel)).tolist()

        contact_set = ContactSet.from_objs_robs(
            [self._contact_pair_modes[mode_id] for mode_id in mode_ids],
            self.objects,
            self.robots,
            additional_constraints=set_force_constraints + self._workspace_constraints,
            additional_base_constraints=self._base_workspace_constraints,
        )

        return contact_set

    def _get_no_rob_contact_objs(self, mode_ids: Iterable[str]):
        # DFS on contact chains to determine if a obj is in contact with a rob.
        adj_list = defaultdict(set)
        all_modes = [self._contact_pair_modes[mode_id] for mode_id in mode_ids]
        modes = [
            mode for mode in all_modes if isinstance(mode, RelaxedInContactPairMode)
        ]
        body_pairs = [(mode.body_a.name, mode.body_b.name) for mode in modes]
        for u, v in body_pairs:
            adj_list[u].add(v)
            adj_list[v].add(u)

        def dfs(v, visited, marked_objs):
            visited.add(v)
            v_body = self._body_dict[v]
            if v_body.mobility_type == MobilityType.UNACTUATED:
                marked_objs.add(v)
            has_robot = v_body.mobility_type == MobilityType.ACTUATED
            for neighbor in adj_list[v]:
                if neighbor not in visited:
                    has_robot = has_robot or dfs(neighbor, visited, marked_objs)
            return has_robot

        visited = set()
        marked_objs = set()
        for v in adj_list:
            if v not in visited:
                component = set()
                has_robot = dfs(v, visited, component)
                if has_robot:
                    marked_objs.update(component)
        all_objs = set([obj.name for obj in self.objects])

        return list(all_objs - marked_objs)

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

    @staticmethod
    def full_to_relaxed_contact_vertex_name(vertex_name: str):
        # Convert string representation of tuple to actual tuple
        tuple_vertex = ast.literal_eval(vertex_name)

        # Initialize dictionaries to store modes for each obj and rob
        res_modes = []

        for mode in tuple_vertex:
            # If mode has obs, then it will be exactly the same
            if "obs" in mode:
                res_modes.append(mode)
                continue

            # Rob-rob modes are not considered in the relaxed contact graph
            if "obj" not in mode:
                continue

            # Convert contact modes to relaxed contact modes
            if "IC" in mode:
                mode = mode.replace("IC|", "RIC|")
            elif "NC" in mode:
                mode = mode.replace("NC|", "RNC|")

            # Remove contact locations from mode
            # This regex will match the undesired parts, like _f1, _v2, or their combinations with other characters
            pattern = r"(_f\d+|_v\d+|\_f\d+_\w+|\_v\d+_\w+)"
            mode = re.sub(pattern, "", mode)
            res_modes.append(mode)

        return str(tuple(res_modes))
