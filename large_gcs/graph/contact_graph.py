import numpy as np
from itertools import combinations, permutations, product
from typing import List
from pydrake.all import (
    Variables,
    DecomposeLinearExpressions,
    DecomposeAffineExpressions,
    HPolyhedron,
    Formula,
    FormulaKind,
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    Cost,
    Constraint,
    Binding,
    MathematicalProgramResult,
    L2NormCost,
)
from tqdm import tqdm
import time
from multiprocessing import Pool

from large_gcs.contact.contact_pair_mode import (
    ContactPairMode,
    generate_contact_pair_modes,
)
from large_gcs.contact.contact_set import ContactSet
from large_gcs.contact.rigid_body import MobilityType, RigidBody
from large_gcs.geometry.convex_set import ConvexSet
from large_gcs.geometry.point import Point
from large_gcs.graph.cost_factory import create_l2norm_edge_cost
from large_gcs.graph.contact_cost_constraint_factory import ContactCostConstraintFactory
from large_gcs.graph.graph import DefaultGraphCostsConstraints, Graph, Vertex, Edge


class ContactGraph(Graph):
    def __init__(
        self,
        static_obstacles: List[RigidBody],
        unactuated_objects: List[RigidBody],
        actuated_robots: List[RigidBody],
        source_pos_objs: List[np.ndarray],
        source_pos_robs: List[np.ndarray],
        target_pos_objs: List[np.ndarray],
        target_pos_robs: List[np.ndarray],
    ):
        """
        Args:
            static_obstacles: List of static obstacles.
            unactuated_objects: List of unactuated objects.
            actuated_robots: List of actuated robots.
            initial_positions: List of initial positions of.
        """
        # Note: The order of operations in this constructor is important
        self.vertices = {}
        self.edges = {}
        self._source_name = None
        self._target_name = None
        self._default_costs_constraints = None
        self.workspace = None
        self._gcs = GraphOfConvexSets()

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

        self.source_pos = source_pos_objs + source_pos_robs
        self.target_pos = target_pos_objs + target_pos_robs

        self._collect_all_variables()
        sets, set_ids = self._generate_contact_sets()

        # Assign costs and constraints
        cc_factory = ContactCostConstraintFactory(self.vars_pos)
        self._default_costs_constraints = DefaultGraphCostsConstraints(
            vertex_costs=[cc_factory.vertex_cost_position_path_length()],
            vertex_constraints=[],
            edge_costs=[],
            edge_constraints=[cc_factory.edge_constraint_position_continuity()],
        )

        sets += [
            self._create_point_set_from_positions(source_pos_objs, source_pos_robs),
            self._create_point_set_from_positions(target_pos_objs, target_pos_robs),
        ]
        set_ids += ["s", "t"]

        # Add convex sets to graph (Need to do this before generating edges)
        self.add_vertices_from_sets(sets, names=set_ids)
        self.set_source("s")
        self.set_target("t")

        edges = self._generate_contact_graph_edges(set_ids)
        self.add_edges_from_vertex_names(*zip(*edges))

        # Check that the source and target are reachable
        assert (
            len(self.outgoing_edges(self.source_name)) > 0
        ), "Source does not overlap with any other set"
        assert (
            len(self.incoming_edges(self.target_name)) > 0
        ), "Target is not reachable from any other set"

    ### SET & EDGE CREATION ###
    def _create_point_set_from_positions(self, obj_positions, rob_positions):
        """Creates a point set from a list of object positions and robot positions"""
        assert len(obj_positions) == len(self.objects)
        assert len(rob_positions) == len(self.robots)
        positions = np.array(obj_positions + rob_positions)
        # Check that all object and robot positions have the same dimension
        assert len(set([pos.shape[0] for pos in positions])) == 1
        assert positions.shape[1] == self.robots[0].dim
        # Repeat each position for the number of position points per set
        coords = np.repeat(positions, self.robots[0].n_pos_points)
        # Note: if more variables are added to the set, e.g. forces this will need to be updated
        return Point(coords)

    def _generate_contact_graph_edges(self, contact_set_ids: List[str]):
        """Generates all possible edges given a set of contact sets."""
        print("Generating edges...(parallel)")
        with Pool() as pool:
            pairs = list(combinations(contact_set_ids, 2))
            sets = [
                (self.vertices[u].convex_set.set, self.vertices[v].convex_set.set)
                for u, v in pairs
            ]
            intersections = list(
                tqdm(pool.imap(self._check_intersection, sets), total=len(sets))
            )
            edges = []
            for (u, v), intersect in zip(pairs, intersections):
                if intersect:
                    edges.append((u, v))
                    edges.append((v, u))
        print(f"{len(edges)} edges generated")
        return edges

    @staticmethod
    def _check_intersection(args):
        u_set, v_set = args
        return u_set.IntersectsWith(v_set)

    def _generate_contact_sets(self):
        """Generates all possible contact sets given a set of static obstacles, unactuated objects, and actuated robots."""
        static_obstacles = self.obstacles
        unactuated_objects = self.objects
        actuated_robots = self.robots
        body_dict = {
            body.name: body
            for body in static_obstacles + unactuated_objects + actuated_robots
        }
        obs_names = [body.name for body in static_obstacles]
        obj_names = [body.name for body in unactuated_objects]
        rob_names = [body.name for body in actuated_robots]

        print(f"Generating contact sets for {len(body_dict)} bodies...")

        movable = obj_names + rob_names
        static_movable_pairs = list(product(obs_names, movable))
        movable_pairs = list(combinations(movable, 2))
        rigid_body_pairs = static_movable_pairs + movable_pairs

        print(
            f"Generating contact pair modes for {len(rigid_body_pairs)} body pairs..."
        )

        body_pair_to_modes = {
            (body1, body2): generate_contact_pair_modes(
                body_dict[body1], body_dict[body2]
            )
            for body1, body2 in tqdm(rigid_body_pairs)
        }
        print(
            f"Each body pair has on average {np.mean([len(modes) for modes in body_pair_to_modes.values()])} modes"
        )
        body_pair_to_mode_names = {
            (body1, body2): [mode.id for mode in modes]
            for (body1, body2), modes in body_pair_to_modes.items()
        }
        mode_ids_to_mode = {
            mode.id: mode for modes in body_pair_to_modes.values() for mode in modes
        }
        # Each set is the cartesian product of the modes for each object pair
        set_ids = list(product(*body_pair_to_mode_names.values()))

        # all_variables = [body.vars_pos for body in  unactuated_objects+actuated_robots]
        # print(f"all_variables shape {np.array(all_variables).shape}")
        # print(all_variables)

        print(f"Generating contact sets for {len(set_ids)} sets...")

        all_contact_sets = [
            ContactSet([mode_ids_to_mode[mode_id] for mode_id in set_id], self.vars_all)
            for set_id in tqdm(set_ids)
        ]

        print(f"Pruning empty sets...")
        non_empty_sets = [
            contact_set
            for contact_set in tqdm(all_contact_sets)
            if not contact_set.set.IsEmpty()
        ]
        non_empty_set_ids = [str(contact_set.id) for contact_set in non_empty_sets]

        print(
            f"{len(non_empty_sets)} sets remain after removing {len(all_contact_sets) - len(non_empty_sets)} empty sets"
        )

        return non_empty_sets, non_empty_set_ids

    def _collect_all_variables(self):
        self.vars_pos = np.concatenate(
            [body.vars_pos for body in self.objects + self.robots]
        )
        # All the decision variables for a single vertex
        self.vars_all = ContactSet.flatten_set_vars(self.vars_pos)

    @property
    def params(self):
        params = super().params
        params.source = self.source_pos
        params.target = self.target_pos
        return params
