import numpy as np
import re
from dataclasses import dataclass
from itertools import combinations, permutations, product
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FFMpegWriter
from collections import defaultdict
from pydrake.all import (
    Variables,
    DecomposeLinearExpressions,
    HPolyhedron,
    Formula,
    FormulaKind,
    ConvexSet as DrakeConvexSet,
    Point as DrakePoint,
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    Cost,
    Constraint,
    Binding,
    MathematicalProgramResult,
    L2NormCost,
    eq,
    ge,
    Expression,
)
from tqdm import tqdm
import time
from multiprocessing import Pool

from large_gcs.contact.contact_pair_mode import (
    InContactPairMode,
    generate_contact_pair_modes,
)
from large_gcs.contact.contact_set import (
    ContactSet,
    ContactPointSet,
    ContactSetDecisionVariables,
)
from large_gcs.contact.rigid_body import MobilityType, RigidBody
from large_gcs.geometry.point import Point
from large_gcs.graph.contact_cost_constraint_factory import (
    vertex_cost_position_path_length,
    vertex_cost_force_actuation_norm_squared,
    vertex_cost_force_actuation_norm,
    edge_constraint_position_continuity,
    edge_cost_constant,
)
from large_gcs.graph.graph import (
    DefaultGraphCostsConstraints,
    Graph,
    ShortestPathSolution,
    Vertex,
    Edge,
)
from large_gcs.algorithms.search_algorithm import AlgVisParams


@dataclass
class ContactShortestPathSolution:
    vertex_path: List[str]
    # shape: (n_pos, n_bodies, n_base_dim)
    pos_trajs: np.ndarray
    # Maps the n_pos index to the index in vertex_path
    pos_transition_map: Dict[int, int] = None


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
        workspace: np.ndarray = None,
        vertex_exclusion: List[str] = None,
        vertex_inclusion: List[str] = None,
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
        self.workspace = workspace
        self.vars = None
        self.obstacles = None
        self.objects = None
        self.robots = None
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

        sets, set_ids = self._generate_contact_sets(vertex_exclusion, vertex_inclusion)

        sets += [
            ContactPointSet(
                "source", self.objects, self.robots, source_pos_objs, source_pos_robs
            ),
            ContactPointSet(
                "target", self.objects, self.robots, target_pos_objs, target_pos_robs
            ),
        ]
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

        edges = self._generate_contact_graph_edges(set_ids)
        self.add_edges_from_vertex_names(
            *zip(*edges),
            costs=self._create_edge_costs(edges),
            constraints=self._create_edge_constraints(edges),
        )

        # Check that the source and target are reachable
        assert (
            len(self.outgoing_edges(self.source_name)) > 0
        ), "Source does not overlap with any other set"
        assert (
            len(self.incoming_edges(self.target_name)) > 0
        ), "Target is not reachable from any other set"

    ### VERTEX AND EDGE COSTS AND CONSTRAINTS ###
    def _create_vertex_costs(self, sets: List[ContactSet]) -> List[List[Cost]]:
        costs = [
            [
                vertex_cost_position_path_length(set.vars),
                vertex_cost_force_actuation_norm(set.vars),
            ]
            if not isinstance(set, ContactPointSet)
            else []
            for set in sets
        ]
        return costs

    def _create_vertex_constraints(
        self, sets: List[ContactSet]
    ) -> List[List[Constraint]]:
        return [[] for set in sets]

    def _create_edge_costs(self, edges: List[Tuple[str, str]]) -> List[List[Cost]]:
        return [
            [
                edge_cost_constant(
                    self.vertices[u].convex_set.vars, self.vertices[v].convex_set.vars
                )
            ]
            for u, v in edges
        ]

    def _create_edge_constraints(
        self, edges: List[Tuple[str, str]]
    ) -> List[List[Constraint]]:
        constraints = []
        for u, v in edges:
            constraints.append(
                [
                    edge_constraint_position_continuity(
                        self.vertices[u].convex_set.vars,
                        self.vertices[v].convex_set.vars,
                    )
                ]
            )
        return constraints

    ### SET & EDGE CREATION ###

    def _generate_contact_graph_edges(
        self, contact_set_ids: List[str]
    ) -> List[Tuple[str, str]]:
        """Generates all possible edges given a set of contact sets."""
        print("Generating edges...(parallel)")
        with Pool() as pool:
            pairs = list(combinations(contact_set_ids, 2))
            sets = [
                (
                    self.vertices[u].convex_set.base_set,
                    self.vertices[v].convex_set.base_set,
                )
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

    def _generate_contact_sets(
        self, vertex_exlcusion: List[str] = None, vertex_inclusion: List[str] = None
    ) -> Tuple[List[ContactSet], List[str]]:
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

        # Set force constraints
        set_force_constraints_dict = defaultdict(list)

        for set_id in set_ids:
            # Add force constraints for each movable body
            for body_name in movable:
                set_force_constraints_dict[set_id] += body_dict[body_name].constraints

            # Collect the forces acting on each body
            body_force_sums = defaultdict(
                lambda: np.full((self.base_dim,), Expression())
            )
            for mode_id in set_id:
                mode = mode_ids_to_mode[mode_id]
                if isinstance(mode, InContactPairMode):
                    body_force_sums[mode.body_a.name] += (
                        -mode.unit_normal * mode.vars_force_mag_BA
                    )
                    body_force_sums[mode.body_b.name] += (
                        mode.unit_normal * mode.vars_force_mag_AB
                    )

            for body_name in movable:
                body = body_dict[body_name]
                if body.mobility_type == MobilityType.ACTUATED:
                    body_force_sums[body_name] += body.vars_force_act
                set_force_constraints_dict[set_id].extend(
                    eq(body.vars_force_res, body_force_sums[body_name]).tolist()
                )

        print(f"Generating contact sets for {len(set_ids)} sets...")

        all_contact_sets = [
            ContactSet(
                [mode_ids_to_mode[mode_id] for mode_id in set_id],
                set_force_constraints_dict[set_id],
                self.objects,
                self.robots,
            )
            for set_id in tqdm(set_ids)
        ]

        print(f"Pruning empty sets...")
        sets_to_keep = [
            contact_set
            for contact_set in tqdm(all_contact_sets)
            if not contact_set.set.IsEmpty()
        ]

        print(
            f"{len(sets_to_keep)} sets remain after removing {len(all_contact_sets) - len(sets_to_keep)} empty sets"
        )

        if vertex_exlcusion is not None:
            print(f"Removing sets matching exclusion strings {vertex_exlcusion}")
            sets_to_keep = [
                contact_set
                for contact_set in tqdm(sets_to_keep)
                if not any(
                    vertex_exlcusion in str(contact_set.id)
                    for vertex_exlcusion in vertex_exlcusion
                )
            ]
            print(f"{len(sets_to_keep)} sets remain after removing excluded sets")

        if vertex_inclusion is not None:
            print(f"Filtering sets for inclusion strings {vertex_inclusion}")
            sets_to_keep = [
                contact_set
                for contact_set in tqdm(sets_to_keep)
                if any(
                    vertex_inclusion in str(contact_set.id)
                    for vertex_inclusion in vertex_inclusion
                )
            ]
            print(f"{len(sets_to_keep)} sets remain after filtering for inclusion sets")

        sets_to_keep_ids = [str(contact_set.id) for contact_set in sets_to_keep]
        return sets_to_keep, sets_to_keep_ids

    def _post_solve(self, sol):
        """Post solve hook that is called after solving by the base graph class"""

        self.contact_spp_sol = self.create_contact_spp_sol(
            sol.vertex_path, sol.ambient_path
        )

    @property
    def params(self):
        params = super().params
        params.source = self.source_pos
        params.target = self.target_pos
        return params

    def create_contact_spp_sol(self, vertex_path, ambient_path):
        """An ambient path is a list of vertices in the higher dimensional space"""
        pos_transition_map = {}
        pos_list = []
        for i, x in enumerate(ambient_path):
            x_vars = self.vertices[vertex_path[i]].convex_set.vars
            x_pos = x_vars.pos_from_all(x)
            # shape: (n_pos, base_dim, n_bodies)
            pos_transition_map[len(pos_list)] = i
            pos_list.extend(x_pos.T.tolist())

        # reshapes pos_list to (n_pos, n_bodies, base_dim)
        pos_trajs = np.array(pos_list).transpose(0, 2, 1)

        return ContactShortestPathSolution(
            vertex_path,
            pos_trajs,
            pos_transition_map,
        )

    def plot_samples_in_set(self, set_name: str, n_samples: int = 100, **kwargs):
        """Plots a single set"""
        options = {"facecolor": "mintcream", "edgecolor": "k", "zorder": 1}
        options.update(kwargs)
        plt.axis("equal")
        vertex = self.vertices[set_name]
        samples = vertex.convex_set.get_samples(n_samples)
        # print(samples)
        raise NotImplementedError
        # Need to modify decompose_ambient_path
        obj_pos_trajectories, rob_pos_trajectories = self.decompose_ambient_path(
            samples
        )

        for j in range(len(samples)):
            # Plot object trajectories
            for i in range(obj_pos_trajectories.shape[0]):
                self.objects[i].plot_at_position(obj_pos_trajectories[i, j])
                # print(f"obj_pos: {obj_pos_trajectories[i, j]}")
            for i in range(rob_pos_trajectories.shape[0]):
                self.robots[i].plot_at_position(rob_pos_trajectories[i, j])
                # print(f"rob_pos: {rob_pos_trajectories[i, j]}")
        for obs in self.obstacles:
            obs.plot()

    def plot_sets(self):
        raise NotImplementedError("Not sure how to visualize high dimensional sets")

    def plot_set_labels(self):
        raise NotImplementedError("Not sure how to visualize high dimensional sets")

    def plot_edges(self):
        raise NotImplementedError("Not sure how to visualize high dimensional sets")

    def plot_path(self):
        assert self.contact_spp_sol is not None, "Must solve before plotting"
        assert self.base_dim == 2, "Can only plot 2D paths"
        sol = self.contact_spp_sol
        plt.figure()
        n_steps = sol.pos_trajs.shape[0]
        for i in range(n_steps):
            obj_pos = sol.pos_trajs[i, : self.n_objects]
            rob_pos = sol.pos_trajs[i, self.n_objects :]
            plt.scatter(*obj_pos.T, marker="+", color=cm.rainbow(i / n_steps))
            plt.scatter(*rob_pos.T, marker=".", color=cm.rainbow(i / n_steps))

        # Add a color bar
        sm = plt.cm.ScalarMappable(
            cmap=cm.rainbow, norm=plt.Normalize(vmin=0, vmax=n_steps)
        )

        plt.colorbar(sm)
        plt.axis("equal")
        # Show the plot
        plt.grid()
        plt.show()

    def animate_solution(self):
        import matplotlib.patches as patches
        import matplotlib.animation as animation

        fig = plt.figure()
        ax = plt.axes(xlim=self.workspace[0], ylim=self.workspace[1])
        ax.set_aspect("equal")
        # Process position trajectories
        trajs, transition_map = self._interpolate_positions(self.contact_spp_sol)

        bodies = self.objects + self.robots

        # Plot static obstacles
        for obs in self.obstacles:
            obs.plot()

        # Plot goal positions
        for i, body in enumerate(bodies):
            body.plot_at_position(self.target_pos[i], color="lightgreen")

        label_text = [body.name for body in bodies]

        polygons = [
            patches.Polygon(
                body.geometry.vertices,
                color="lightblue"
                if body.mobility_type == MobilityType.UNACTUATED
                else "lightsalmon",
            )
            for body in bodies
        ]
        poly_offset = [
            poly.get_xy() - body.geometry.center for poly, body in zip(polygons, bodies)
        ]
        labels = [
            ax.text(*body.geometry.center, label, ha="center", va="center")
            for body, label in zip(bodies, label_text)
        ]
        vertex_annotation = ax.annotate(
            transition_map[0],
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
        )
        for poly in polygons:
            ax.add_patch(poly)

        # force_res_quivers = [ax.quiver([], [], [], []) for _ in range(len(bodies))]
        # force_res_vals = np.concatenate(
        #     (
        #         self.contact_spp_sol.object_force_res_trajectories,
        #         self.contact_spp_sol.robot_force_res_trajectories,
        #     )
        # )

        def animate(i):
            for j in range(len(bodies)):
                polygons[j].set_xy(poly_offset[j] + trajs[i][j])
                labels[j].set_position(trajs[i][j])
                if i in transition_map:
                    # force_res_quivers[j].set_offsets(poly_offset[j] + trajs[i][j])
                    # force_res_quivers[j].set_UVC(*force_res_vals[j][transition_map[i]])
                    vertex_annotation.set_text(
                        self.contact_spp_sol.vertex_path[transition_map[i]]
                    )
                else:
                    pass
                    # force_res_quivers[j].set_offsets(poly_offset[j] + trajs[i][j])
            return polygons

        anim = animation.FuncAnimation(
            fig, animate, frames=trajs.shape[0], interval=50, blit=True
        )
        return anim

    @staticmethod
    def _interpolate_positions(contact_sol, max_gap: float = 0.1):
        # Input has shape (n_movable bodies, n_sets_in_path, n_pos_per_set, n_base_dim) OLD
        # Input has shape (n_pos, n_bodies, n_base_dim)
        trajs_in = contact_sol.pos_trajs
        # print(f"trajs_in shape {trajs_in.shape}")
        transition_map = {}
        # Final list is going to have shape (n_pos, n_movable bodies (objects then robots), n_base_dim)
        trajs_out = []

        # Add in the first position
        trajs_out.append(trajs_in[0])
        transition_map[0] = 0
        for n_pos in range(1, trajs_in.shape[0]):
            if n_pos in contact_sol.pos_transition_map:
                transition_map[len(trajs_out)] = contact_sol.pos_transition_map[n_pos]
            # Loop over all the bodies
            m_gaps = np.linalg.norm(trajs_in[n_pos] - trajs_in[n_pos - 1], axis=1)
            m_gap = np.max(m_gaps)
            # If the gap is larger than the max gap, interpolate
            if m_gap > max_gap:
                # Number of segments for interpolation
                num_segments = int(np.ceil(m_gap / max_gap))
                # Generate interpolated positions
                for j in range(1, num_segments):
                    interp_pos = (j / num_segments) * (
                        trajs_in[n_pos] - trajs_in[n_pos - 1]
                    ) + trajs_in[n_pos - 1]
                    trajs_out.append(interp_pos)
            else:
                trajs_out.append(trajs_in[n_pos])

        return np.array(trajs_out), transition_map

    @property
    def n_obstacles(self):
        return len(self.obstacles)

    @property
    def n_objects(self):
        return len(self.objects)

    @property
    def n_robots(self):
        return len(self.robots)

    @property
    def base_dim(self):
        return self.robots[0].dim
