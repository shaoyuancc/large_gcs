import ast
import logging
from collections import defaultdict
from copy import copy
from itertools import combinations, product
from multiprocessing import Pool
from typing import List

import numpy as np
from pydrake.all import Cost
from tqdm import tqdm

from large_gcs.contact.contact_regions_set import ContactRegionParams, ContactRegionsSet
from large_gcs.contact.contact_set import ContactPointSet, ContactSet
from large_gcs.contact.contact_set_decision_variables import ContactSetDecisionVariables
from large_gcs.contact.rigid_body import MobilityType, RigidBody
from large_gcs.geometry.polyhedron import Polyhedron
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
        should_incl_simul_mode_switches: bool = True,
        should_add_gcs: bool = False,
        should_add_const_edge_cost: bool = True,
    ):
        """
        Can either specify target_pos or target_region_params, but not both.
        include_simultaneous_mode_switches determines whether or not to include simultaneous mode switches as neighbors.
        add_gcs determines whether or not to add the drake gcs vertices and edges to the graph.
        It is not required if you are only using the incremental graph as a reference but not actually
        solving any gcs problems on it directly.
        """
        Graph.__init__(self, workspace=workspace)
        assert self.workspace is not None, "Workspace must be set"
        # Note: The order of operations in this constructor is important

        self.target_pos = None
        self.target_region_params = None
        self.target_regions = None
        self._should_incl_simul_mode_switches = should_incl_simul_mode_switches
        self._should_add_gcs = should_add_gcs
        self._should_add_const_edge_cost = should_add_const_edge_cost

        if not should_add_const_edge_cost:

            def _create_single_empty_edge_cost(u: str, v: str) -> List[Cost]:
                return []

            self._create_single_edge_costs = _create_single_empty_edge_cost

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

        body_name_to_source_pos = {}
        for i, body in enumerate(self._movable):
            body_name_to_source_pos[body] = self.source_pos[i]

        if self.target_pos is not None:
            body_name_to_target_pos = {}
            for i, body in enumerate(self._movable):
                body_name_to_target_pos[body] = self.target_pos[i]

        elif self.target_region_params is not None:
            body_name_to_indiv_target_region_params = {}
            self._bodies_w_target_region = set()
            for params in self.target_region_params:
                individual_params = copy(params)
                if params.obj_indices is not None:
                    for body_index in params.obj_indices:
                        individual_params.obj_indices = [0]
                        individual_params.rob_indices = None
                        body_name_to_indiv_target_region_params[
                            self.objects[body_index].name
                        ] = individual_params
                        self._bodies_w_target_region.add(self.objects[body_index].name)
                if params.rob_indices is not None:
                    for body_index in params.rob_indices:
                        individual_params.rob_indices = [0]
                        individual_params.obj_indices = None
                        body_name_to_indiv_target_region_params[
                            self.robots[body_index].name
                        ] = individual_params
                        self._bodies_w_target_region.add(self.robots[body_index].name)

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
                if body.mobility_type == MobilityType.STATIC:
                    continue
                additional_constraints.extend(body.base_workspace_constraints)
                movable_body_name = body_name
                if body.mobility_type == MobilityType.UNACTUATED:
                    objs.append(body)
                elif body.mobility_type == MobilityType.ACTUATED:
                    robs.append(body)

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
            found_neighbor = False
            for id in mode_ids:
                if base_polyhedra[id].PointInSet(source_pos):
                    found_neighbor = True
                    source_neighbor_contact_pair_modes.append(id)
                    break  # We are just looking for a single neighbor
            assert (
                found_neighbor
            ), f"Could not find source contact pair neighbor for {body_pair}"

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

        assert len(self._body_pair_to_mode_ids.keys()) == len(
            source_neighbor_contact_pair_modes
        ), "Should have a single contact pair mode for each "
        self._generate_neighbor(
            u=self.source_name,
            v=str(tuple(source_neighbor_contact_pair_modes)),
            is_v_in_vertices=False,
            v_set=self._create_contact_set_from_contact_pair_mode_ids(
                source_neighbor_contact_pair_modes
            ),
        )

    def solve_shortest_path(self, use_convex_relaxation=False) -> ShortestPathSolution:
        if self._should_add_gcs:
            return super().solve_shortest_path(use_convex_relaxation)
        else:
            raise ValueError(
                f"Incremental graph should_add_gcs is False. Must set to True in order to solve."
            )

    def solve_convex_restriction(self, active_edges: List[str]) -> ShortestPathSolution:
        if self._should_add_gcs:
            return super().solve_convex_restriction(active_edges)
        else:
            raise ValueError(
                f"Incremental graph should_add_gcs is False. Must set to True in order to solve."
            )

    def solve_factored_shortest_path(
        self, transition: str, targets: List[str], use_convex_relaxation=False
    ) -> ShortestPathSolution:
        if self._should_add_gcs:
            return super().solve_factored_shortest_path(
                transition, targets, use_convex_relaxation
            )
        else:
            raise ValueError(
                f"Incremental graph should_add_gcs is False. Must set to True in order to solve."
            )

    def solve_factored_partial_convex_restriction(
        self, active_edges: List[str], transition: str, targets: List[str]
    ) -> ShortestPathSolution:
        if self._should_add_gcs:
            return super().solve_factored_partial_convex_restriction(
                active_edges, transition, targets
            )
        else:
            raise ValueError(
                f"Incremental graph should_add_gcs is False. Must set to True in order to solve."
            )

    def generate_neighbors(self, u_vertex_name: str) -> None:
        """Generates neighbors and adds them to the graph, also adds edges from vertex to neighbors"""
        if u_vertex_name == self.source_name:
            # We already have the neighbors of the source vertex
            return
        elif u_vertex_name == self.target_name:
            raise ValueError("Should not need to generate neighbors for target vertex")

        # Convert string representation of tuple to actual tuple
        mode_ids = ast.literal_eval(u_vertex_name)

        if self._should_incl_simul_mode_switches:
            mode_ids_for_each_body_pair = []
            for i, id in enumerate(mode_ids):
                mode_ids_for_each_body_pair.append(
                    # This order is what allows us to remove the first element in set_ids
                    [id]
                    + self._adj_modes[id]
                )
            set_ids = list(product(*mode_ids_for_each_body_pair))
            # Remove the first entry in the list which would be the current set
            set_ids = set_ids[1:]

        else:
            # Flip the each mode through all possible adjacent modes (Only flip one mode at a time)
            # This excludes multiple simultaneous flips, which are technically also valid neighbors
            # but we are not considering them for now.
            set_ids = []
            for i, id in enumerate(mode_ids):
                for adj_id in self._adj_modes[id]:
                    neighbor_set_id = list(copy(mode_ids))
                    neighbor_set_id[i] = adj_id
                    set_ids.append(tuple(neighbor_set_id))
        potential_neighbors = []
        neighbor_generator_inputs = []
        for set_id in set_ids:
            v_name = str(set_id)
            if (u_vertex_name, v_name) in self.edges:
                # vertex and edge already exits, do nothing.
                logger.debug(
                    f"vertex and edge already exist for {u_vertex_name} -> {v_name}"
                )
                continue

            if v_name in self.vertices:
                is_v_in_vertices = True
                v_set = self.vertices[v_name].convex_set
            else:
                is_v_in_vertices = False
                v_set = self._create_contact_set_from_contact_pair_mode_ids(set_id)
            potential_neighbors.append(
                (self.vertices[u_vertex_name].convex_set.base_set, v_set.base_set)
            )
            neighbor_generator_inputs.append(
                (u_vertex_name, v_name, is_v_in_vertices, v_set)
            )

        if self._does_vertex_have_possible_edge_to_target(u_vertex_name):
            potential_neighbors.append(
                (
                    self.vertices[u_vertex_name].convex_set.base_set,
                    self.vertices[self.target_name].convex_set.base_set,
                )
            )
            neighbor_generator_inputs.append(
                (
                    u_vertex_name,
                    self.target_name,
                    True,
                    self.vertices[self.target_name].convex_set,
                )
            )

        # Parallelization seems to add significant overhead and the above code is already very slow when there are many potential neighbors
        # with Pool() as pool:
        #     logger.debug(f"Generating {len(set_ids)} possible neighbors for {u_vertex_name}")
        #     intersections = list(
        #         tqdm(pool.imap(self._check_intersection, potential_neighbors), total=len(potential_neighbors))
        #     )
        intersections = [
            self._check_intersection(potential_neighbor)
            for potential_neighbor in potential_neighbors
        ]

        for inputs, intersect in zip(neighbor_generator_inputs, intersections):
            if intersect:
                self._generate_neighbor(*inputs)

    def _generate_neighbor(
        self, u: str, v: str, is_v_in_vertices: bool, v_set: ContactSet
    ) -> None:
        if not is_v_in_vertices:
            vertex = Vertex(
                v_set,
                costs=self._create_single_vertex_costs(v_set),
                constraints=self._create_single_vertex_constraints(v_set),
            )
            self.add_vertex(vertex, v, should_add_to_gcs=self._should_add_gcs)
        self.add_edge(
            Edge(
                u=u,
                v=v,
                costs=self._create_single_edge_costs(u, v),
                constraints=self._create_single_edge_constraints(u, v),
            ),
            should_add_to_gcs=self._should_add_gcs,
        )

    def _does_vertex_have_possible_edge_to_target(self, vertex_name: str) -> bool:
        # Determine if we can add an edge to the target vertex
        # Convert string representation of tuple to actual tuple
        mode_ids = ast.literal_eval(vertex_name)
        possible_edge_to_target = []
        if self.target_pos is not None:
            for id in mode_ids:
                mode = self._contact_pair_modes[id]
                # Only consider body pairs that are not both movable
                # Note for target pos, all movable bodies have a position set for them
                if not (
                    mode.body_a.mobility_type != MobilityType.STATIC
                    and mode.body_b.mobility_type != MobilityType.STATIC
                ):
                    possible_edge_to_target.append(
                        id in self._modes_w_possible_edge_to_target
                    )
        else:
            for id in mode_ids:
                mode = self._contact_pair_modes[id]
                # Only consider body pairs that are not both movable AND
                # where the movable body has a target set for it
                if not (
                    mode.body_a.mobility_type != MobilityType.STATIC
                    and mode.body_b.mobility_type != MobilityType.STATIC
                ) and (
                    mode.body_a.name in self._bodies_w_target_region
                    or mode.body_b.name in self._bodies_w_target_region
                ):
                    logger.debug(f"considering {id}")
                    possible_edge_to_target.append(
                        id in self._modes_w_possible_edge_to_target
                    )
        if all(possible_edge_to_target):
            return True
        return False

    ### SERIALIZATION METHODS ###

    def save_to_file(self, path: str):
        if self.target_pos is None:
            target_pos_objs = None
            target_pos_robs = None
        else:
            target_pos_objs = self.target_pos[: self.n_objects]
            target_pos_robs = self.target_pos[self.n_objects :]
        np.save(
            path,
            {
                "obs_params": [obs.params for obs in self.obstacles],
                "objs_params": [obj.params for obj in self.objects],
                "robs_params": [rob.params for rob in self.robots],
                "source_pos_objs": self.source_pos[: self.n_objects],
                "source_pos_robs": self.source_pos[self.n_objects :],
                "target_pos_objs": target_pos_objs,
                "target_pos_robs": target_pos_robs,
                "target_region_params": self.target_region_params,
                "workspace": self.workspace,
            },
        )

    @classmethod
    def load_from_file(
        cls,
        path: str,
        **kwargs,
    ):
        data = np.load(path, allow_pickle=True).item()
        obs = [RigidBody.from_params(params) for params in data["obs_params"]]
        objs = [RigidBody.from_params(params) for params in data["objs_params"]]
        robs = [RigidBody.from_params(params) for params in data["robs_params"]]
        if "target_region_params" in data:
            target_region_params = data["target_region_params"]
        else:
            target_region_params = None

        cg = cls(
            static_obstacles=obs,
            unactuated_objects=objs,
            actuated_robots=robs,
            source_pos_objs=data["source_pos_objs"],
            source_pos_robs=data["source_pos_robs"],
            target_pos_objs=data["target_pos_objs"],
            target_pos_robs=data["target_pos_robs"],
            target_region_params=target_region_params,
            workspace=data["workspace"],
            **kwargs,
        )
        return cg
