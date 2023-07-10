import numpy as np
from dataclasses import dataclass
from itertools import combinations, permutations, product
from typing import List
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
from large_gcs.contact.contact_set import ContactSet, ContactSetDecisionVariables
from large_gcs.contact.rigid_body import MobilityType, RigidBody
from large_gcs.geometry.convex_set import ConvexSet
from large_gcs.geometry.point import Point
from large_gcs.graph.cost_factory import create_l2norm_edge_cost
from large_gcs.graph.contact_cost_constraint_factory import ContactCostConstraintFactory
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
    # shape: (n_objects, n_sets_in_path, n_pos_per_set, n_base_dimension)
    object_pos_trajectories: np.ndarray
    # shape: (n_robots, n_sets_in_path, n_pos_per_set, n_base_dimension)
    robot_pos_trajectories: np.ndarray
    vertex_path: List[str]


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

        sets, set_ids = self._generate_contact_sets()

        # Assign costs and constraints
        cc_factory = ContactCostConstraintFactory(self.vars)
        self._default_costs_constraints = DefaultGraphCostsConstraints(
            vertex_costs=[
                cc_factory.vertex_cost_position_path_length(),
                # cc_factory.vertex_cost_position_path_length_squared(),
                cc_factory.vertex_cost_force_actuation_norm_squared(),
            ],
            vertex_constraints=[],
            edge_costs=[
                cc_factory.edge_cost_constant(),
                # cc_factory.edge_costs_position_continuity_norm(),
            ],
            edge_constraints=[
                cc_factory.edge_constraint_position_continuity(),
                # cc_factory.edge_constraint_position_continuity_linearconstraint(),
            ],
        )
        self.cc_factory = cc_factory

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
        assert len(obj_positions) == self.n_objects
        assert len(rob_positions) == self.n_robots
        positions = np.array(obj_positions + rob_positions)
        # Check that all object and robot positions have the same dimension
        assert len(set([pos.shape[0] for pos in positions])) == 1
        assert positions.shape[1] == self.robots[0].dim
        # Repeat each position for the number of position points per set
        pos_repeated_flattened = np.repeat(
            positions, self.robots[0].n_pos_points
        ).flatten()
        # Assume all other decision variables are zero (forces)
        coords = np.pad(
            pos_repeated_flattened,
            (0, self.vars.all.size - pos_repeated_flattened.size),
            mode="constant",
        )
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
        in_contact_pair_modes = [
            mode
            for mode in mode_ids_to_mode.values()
            if isinstance(mode, InContactPairMode)
        ]

        # print(f"in_contact_pair_modes: {np.array([mode.id for mode in in_contact_pair_modes])}")

        self.vars = ContactSetDecisionVariables(
            self.objects, self.robots, in_contact_pair_modes
        )

        # Each set is the cartesian product of the modes for each object pair
        set_ids = list(product(*body_pair_to_mode_names.values()))

        # Set force constraints
        set_force_constraints_dict = defaultdict(list)
        eps = 1e-3

        for set_id in set_ids:
            for in_contact_mode in in_contact_pair_modes:
                # Enforce the active forces to be greater than a small constant
                if in_contact_mode.id in set_id:
                    # If bodies A and B are in contact, A must be exerting some positive force on B, and vice versa
                    set_force_constraints_dict[set_id] += [
                        ge(in_contact_mode.vars_force_mag_AB, eps).item(),
                        ge(in_contact_mode.vars_force_mag_BA, eps).item(),
                    ]
                else:
                    # Bodies not incontact must not exert any force on each other
                    set_force_constraints_dict[set_id] += [
                        eq(in_contact_mode.vars_force_mag_AB, 0).item(),
                        eq(in_contact_mode.vars_force_mag_BA, 0).item(),
                    ]

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
                set_force_constraints_dict[set_id].extend(
                    eq(
                        body_dict[body_name].vars_force_res, body_force_sums[body_name]
                    ).tolist()
                )

        print(f"Generating contact sets for {len(set_ids)} sets...")

        all_contact_sets = [
            ContactSet(
                [mode_ids_to_mode[mode_id] for mode_id in set_id],
                set_force_constraints_dict[set_id],
                self.vars.all,
            )
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

    def _post_solve(self, sol):
        """Post solve hook that is called after solving by the base graph class"""
        vertex_path, ambient_path = zip(*sol.path)
        obj_pos_trajectories, rob_pos_trajectories = self.decompose_ambient_path(
            ambient_path
        )

        self.contact_spp_sol = ContactShortestPathSolution(
            obj_pos_trajectories, rob_pos_trajectories, vertex_path
        )

    @property
    def params(self):
        params = super().params
        params.source = self.source_pos
        params.target = self.target_pos
        return params

    def decompose_ambient_path(self, ambient_path):
        """An ambient path is a list of vertices in the higher dimensional space"""
        n_pos_per_set = self.vars.pos.shape[2]
        obj_pos_trajectories = np.zeros(
            (self.n_objects, len(ambient_path), n_pos_per_set, self.base_dim)
        )
        rob_pos_trajectories = np.zeros(
            (self.n_robots, len(ambient_path), n_pos_per_set, self.base_dim)
        )
        for i, x in enumerate(ambient_path):
            x = ContactSetDecisionVariables.vars_pos_from_vars_all(self.vars.pos, x)
            # print(f"x shape {x.shape}")
            # print(x)

            x_objs_pos_transposed = np.transpose(x[: self.n_objects], (0, 2, 1))
            obj_pos_trajectories[:, i, :, :] = x_objs_pos_transposed
            x_robs_pos_transposed = np.transpose(x[self.n_objects :], (0, 2, 1))
            rob_pos_trajectories[:, i, :, :] = x_robs_pos_transposed
        return obj_pos_trajectories, rob_pos_trajectories

    def plot_samples_in_set(self, set_name: str, n_samples: int = 100, **kwargs):
        """Plots a single set"""
        options = {"facecolor": "mintcream", "edgecolor": "k", "zorder": 1}
        options.update(kwargs)
        plt.axis("equal")
        vertex = self.vertices[set_name]
        samples = vertex.convex_set.get_samples(n_samples)
        # print(samples)
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
        # Create a new figure
        plt.figure()
        if sol.robot_pos_trajectories.size > 0:
            traj = np.reshape(
                sol.robot_pos_trajectories,
                (
                    sol.robot_pos_trajectories.shape[0],
                    -1,
                    sol.robot_pos_trajectories.shape[-1],
                ),
            )
            n_time_steps = traj.shape[1]

            # Create a color map
            colors = cm.rainbow(np.linspace(0, 1, n_time_steps))
            # Add a color bar
            sm = plt.cm.ScalarMappable(
                cmap=cm.rainbow, norm=plt.Normalize(vmin=0, vmax=n_time_steps)
            )

            # Plot robot trajectories
            for i in range(traj.shape[0]):
                for j in range(n_time_steps):
                    plt.scatter(*traj[i, j], color=colors[j])

        if sol.object_pos_trajectories.size > 0:
            traj = np.reshape(
                sol.object_pos_trajectories,
                (
                    sol.object_pos_trajectories.shape[0],
                    -1,
                    sol.object_pos_trajectories.shape[-1],
                ),
            )
            # Plot object trajectories
            for i in range(traj.shape[0]):
                for j in range(n_time_steps):
                    plt.scatter(*traj[i, j], color=colors[j])

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
        # remove duplicate positions
        trajs, transition_map = self._interpolate_positions(self.contact_spp_sol)

        bodies = self.objects + self.robots
        label_text = [body.name for body in bodies]

        polygons = [patches.Polygon(body.geometry.vertices) for body in bodies]
        poly_offset = [
            poly.get_xy() - body.geometry.center for poly, body in zip(polygons, bodies)
        ]
        labels = [
            ax.text(*body.geometry.center, label)
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

        def animate(i):
            for j in range(len(bodies)):
                polygons[j].set_xy(poly_offset[j] + trajs[i][j])
                labels[j].set_position(trajs[i][j])
                if i in transition_map:
                    vertex_annotation.set_text(transition_map[i])
            return polygons

        # Plot static obstacles
        for obs in self.obstacles:
            obs.plot()

        anim = animation.FuncAnimation(
            fig, animate, frames=trajs.shape[0], interval=50, blit=True
        )
        return anim

    @staticmethod
    def _interpolate_positions(contact_sol, max_gap: float = 0.1):
        # Input has shape (n_movable bodies, n_sets_in_path, n_pos_per_set, n_base_dim)
        trajs_in = np.vstack(
            (contact_sol.object_pos_trajectories, contact_sol.robot_pos_trajectories)
        )
        # print(f"trajs_in shape {trajs_in.shape}")
        transition_map = {}
        # Final list is going to have shape (n_steps, n_movable bodies (objects then robots), n_base_dim)
        trajs_out = []
        # Loop over n_sets_in_path
        for n_set in range(trajs_in.shape[1]):
            # Add in the first position
            trajs_out.append(trajs_in[:, n_set, 0])
            transition_map[len(trajs_out) - 1] = contact_sol.vertex_path[n_set]
            # Loop over n_pos_per_set which is the third index in pos_traj
            for n_pos in range(1, trajs_in.shape[2]):
                # Loop over all the bodies
                m_gap = 0
                for n_body in range(trajs_in.shape[0]):
                    gap = np.linalg.norm(
                        trajs_in[n_body, n_set, n_pos]
                        - trajs_in[n_body, n_set, n_pos - 1]
                    )
                    if gap > m_gap:
                        m_gap = gap
                # If the gap is larger than the max gap, interpolate
                if m_gap > max_gap:
                    # Number of segments for interpolation
                    num_segments = int(np.ceil(m_gap / max_gap))
                    # Generate interpolated positions
                    for j in range(1, num_segments):
                        interp_pos = (j / num_segments) * (
                            trajs_in[:, n_set, n_pos] - trajs_in[:, n_set, n_pos - 1]
                        ) + trajs_in[:, n_set, n_pos - 1]
                        trajs_out.append(interp_pos)
                else:
                    trajs_out.append(trajs_in[:, n_set, n_pos])

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
