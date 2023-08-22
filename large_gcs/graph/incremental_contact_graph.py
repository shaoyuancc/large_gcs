import ast
import logging
from collections import defaultdict
from copy import copy
from itertools import combinations, product
from multiprocessing import Pool
from typing import Iterable, List, Tuple

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
from large_gcs.graph.graph import Edge, Graph, ShortestPathSolution, Vertex

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
        assert self.workspace is not None, "Workspace must be set"
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
        self._initialize_set_generation_variables()

        workspace_pos_constraints = {
            body_name: self._body_dict[body_name].create_workspace_position_constraints(
                self.workspace, base_only=True
            )
            for body_name in self._movable
        }

        body_name_to_source_pos = {}
        for i, body in enumerate(self._movable):
            body_name_to_source_pos[body] = self.source_pos[i]

        if self.target_pos is not None:
            body_name_to_target_pos = {}
            for i, body in enumerate(self._movable):
                body_name_to_target_pos[body] = self.target_pos[i]

        elif self.target_region_params is not None:
            body_name_to_indiv_target_region_params = {}
            for params in self.target_region_params:
                individual_params = copy(params)
                if params.obj_indices is not None:
                    for body_index in params.obj_indices:
                        individual_params.obj_indices = [0]
                        individual_params.rob_indices = None
                        body_name_to_indiv_target_region_params[
                            self.objects[body_index].name
                        ] = individual_params
                if params.rob_indices is not None:
                    for body_index in params.rob_indices:
                        individual_params.rob_indices = [0]
                        individual_params.obj_indices = None
                        body_name_to_indiv_target_region_params[
                            self.robots[body_index].name
                        ] = individual_params

        self._modes_w_possible_edge_to_target = set()

        self._adj_modes = defaultdict(list)
        sets = []
        pairs_of_modes = []
        source_neighbor_contact_pair_modes = []

        for body_pair, mode_ids in self._body_pair_to_mode_ids.items():
            objs = []
            robs = []
            additional_constraints = []

            for body_name in body_pair:
                body = self._body_dict[body_name]
                if body.mobility_type == MobilityType.UNACTUATED:
                    objs.append(body)
                    additional_constraints.extend(workspace_pos_constraints[body_name])
                    movable_body_name = body_name
                elif body.mobility_type == MobilityType.ACTUATED:
                    robs.append(body)
                    additional_constraints.extend(workspace_pos_constraints[body_name])
                    movable_body_name = body_name

            vars = ContactSetDecisionVariables.from_objs_robs(objs, robs)
            base_polyhedra = {
                id: self._contact_pair_modes[id].create_base_polyhedron(
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
            # Determining an outgoing edge for source vertex
            source_pos = np.array([])
            for body_name in body_pair:
                if body_name in body_name_to_source_pos:
                    source_pos = np.append(
                        source_pos, body_name_to_source_pos[body_name]
                    )
            for id in mode_ids:
                if base_polyhedra[id].PointInSet(source_pos):
                    source_neighbor_contact_pair_modes.append(id)
                    break  # We are just looking for a single neighbor

            # Determining incoming edges for target vertex
            if self.target_pos is not None:
                target_pos = np.array([])
                for body_name in body_pair:
                    if body_name in body_name_to_target_pos:
                        target_pos = np.append(
                            target_pos, body_name_to_target_pos[body_name]
                        )
                for id in mode_ids:
                    if base_polyhedra[id].PointInSet(target_pos):
                        self._modes_w_possible_edge_to_target.add(id)
            elif (
                self.target_region_params is not None
                and len(objs) + len(robs) == 1
                and movable_body_name in body_name_to_indiv_target_region_params
            ):
                target_set = ContactRegionsSet(
                    objects=objs,
                    robots=robs,
                    contact_region_params=[
                        body_name_to_indiv_target_region_params[movable_body_name]
                    ],
                    name="target",
                )
                for id in mode_ids:
                    if self._check_intersection(
                        (base_polyhedra[id], target_set.base_set)
                    ):
                        self._modes_w_possible_edge_to_target.add(id)

        with Pool() as pool:
            logger.info(f"Calculating adjacent contact pair modes ({len(sets)})")
            intersections = list(
                tqdm(pool.imap(self._check_intersection, sets), total=len(sets))
            )
            for (mode_id1, mode_id2), intersection in zip(
                pairs_of_modes, intersections
            ):
                if intersection:
                    self._adj_modes[mode_id1].append(mode_id2)
                    self._adj_modes[mode_id2].append(mode_id1)

        assert len(self._body_pair_to_mode_ids) == len(
            source_neighbor_contact_pair_modes
        ), "Should have a single contact pair mode for each "
        self._generate_vertex_neighbor(
            self.source_name, source_neighbor_contact_pair_modes
        )

    def solve_shortest_path(self, use_convex_relaxation=False) -> ShortestPathSolution:
        raise NotImplementedError(
            "Not implemented for incremental contact graph, GCS vertices and edges are not created, inc graph is just meant to be used as a reference."
        )

    def solve_convex_restriction(
        self, active_edges: List[Tuple[str, str]]
    ) -> ShortestPathSolution:
        raise NotImplementedError(
            "Not implemented for incremental contact graph, GCS vertices and edges are not created, inc graph is just meant to be used as a reference."
        )

    def solve_factored_shortest_path(
        self, transition: str, targets: List[str], use_convex_relaxation=False
    ) -> ShortestPathSolution:
        raise NotImplementedError(
            "Not implemented for incremental contact graph, GCS vertices and edges are not created, inc graph is just meant to be used as a reference."
        )

    def solve_factored_partial_convex_restriction(
        self, active_edges: List[Tuple[str, str]], transition: str, targets: List[str]
    ) -> ShortestPathSolution:
        raise NotImplementedError(
            "Not implemented for incremental contact graph, GCS vertices and edges are not created, inc graph is just meant to be used as a reference."
        )

    def generate_neighbors(self, vertex_name: str) -> None:
        """Generates neighbors and adds them to the graph, also adds edges from vertex to neighbors"""
        if vertex_name == self.source_name:
            # We already have the neighbors of the source vertex
            return
        elif vertex_name == self.target_name:
            raise ValueError("Should not need to generate neighbors for target vertex")

        # Convert string representation of tuple to actual tuple
        mode_ids = ast.literal_eval(vertex_name)
        # Flip the each mode through all possible adjacent modes (Only flip one mode at a time)
        # This excludes multiple simultaneous flips, which are technically also valid neighbors
        # but we are not considering them for now.
        possible_edge_to_target = []
        for i, id in enumerate(mode_ids):
            for adj_id in self._adj_modes[id]:
                neighbor_set_id = copy(mode_ids)
                neighbor_set_id[i] = adj_id
                self._generate_vertex_neighbor(vertex_name, neighbor_set_id)

            mode = self._contact_pair_modes[id]
            # Only consider body pairs that are not both movable (that's what our possible edge to target
            # conditions are based on)
            if (
                mode.body_a.mobility_type != MobilityType.STATIC
                and mode.body_b.mobility_type != MobilityType.STATIC
            ):
                possible_edge_to_target.append(
                    id in self._modes_w_possible_edge_to_target
                )

        if all(possible_edge_to_target):
            # Add edge to target vertex
            self.add_edge(
                Edge(
                    u=vertex_name,
                    v=self.target_name,
                    costs=self._create_single_edge_costs(vertex_name, self.target_name),
                    constraints=self._create_single_edge_constraints(
                        vertex_name, self.target_name
                    ),
                ),
                add_to_gcs=False,
            )

    def _generate_vertex_neighbor(
        self, u: str, v_contact_pair_mode_ids: Iterable[str]
    ) -> None:
        """Assumes that u is already a vertex in the graph."""
        vertex_name = str(tuple(v_contact_pair_mode_ids))

        if (u, vertex_name) in self.edges:
            # vertex and edge already exits, do nothing.
            return

        if vertex_name not in self.vertices:
            v_set = self._create_contact_set_from_contact_pair_mode_ids(
                v_contact_pair_mode_ids
            )
            vertex = Vertex(
                v_set,
                costs=self._create_single_vertex_costs(v_set),
                constraints=self._create_single_vertex_constraints(v_set),
            )
            self.add_vertex(vertex, vertex_name, add_to_gcs=False)

        self.add_edge(
            Edge(
                u=u,
                v=vertex_name,
                costs=self._create_single_edge_costs(u, vertex_name),
                constraints=self._create_single_edge_constraints(u, vertex_name),
            ),
            add_to_gcs=False,
        )
