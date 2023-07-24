import logging
from collections import defaultdict
from dataclasses import dataclass
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
from large_gcs.contact.contact_set import ContactPointSet, ContactSet
from large_gcs.contact.rigid_body import BodyColor, MobilityType, RigidBody
from large_gcs.graph.contact_cost_constraint_factory import (
    edge_constraint_position_continuity,
    edge_cost_constant,
    vertex_cost_force_actuation_norm,
    vertex_cost_position_path_length,
)
from large_gcs.graph.graph import Graph, ShortestPathSolution

logger = logging.getLogger(__name__)


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
        # Note: The order of operations in this constructor is important
        self.vertices = {}
        self.edges = {}
        self._source_name = None
        self._target_name = None
        self._default_costs_constraints = None
        self.workspace = workspace
        self.vertex_inclusion = vertex_inclusion
        self.vertex_exclusion = vertex_exclusion
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
        sets, set_ids = self._generate_contact_sets(
            contact_pair_modes, contact_set_mode_ids, vertex_exclusion, vertex_inclusion
        )

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
        if edge_keys is None:
            edges = self._generate_contact_graph_edges(set_ids)
        else:
            edges = self._filter_edge_keys(edge_keys)
        self.add_edges_from_vertex_names(
            *zip(*edges),
            costs=self._create_edge_costs(edges),
            constraints=self._create_edge_constraints(edges),
        )

        # Check that the source and target are reachable
        if len(self.outgoing_edges(self.source_name)) == 0:
            logger.warn("Source does not overlap with any other set")
        if len(self.incoming_edges(self.target_name)) == 0:
            logger.warn("Target is not reachable from any other set")

        logger.info(f"Created contact graph: {self.params}")

    ### VERTEX AND EDGE COSTS AND CONSTRAINTS ###
    def _create_vertex_costs(self, sets: List[ContactSet]) -> List[List[Cost]]:
        logger.info("Creating vertex costs...")
        costs = [
            [
                vertex_cost_position_path_length(set.vars),
                vertex_cost_force_actuation_norm(set.vars),
            ]
            if not isinstance(set, ContactPointSet)
            else []
            for set in tqdm(sets)
        ]
        return costs

    def _create_vertex_constraints(
        self, sets: List[ContactSet]
    ) -> List[List[Constraint]]:
        return [[] for set in sets]

    def _create_edge_costs(self, edges: List[Tuple[str, str]]) -> List[List[Cost]]:
        logger.info("Creating edge costs...")
        return [
            [
                edge_cost_constant(
                    self.vertices[u].convex_set.vars, self.vertices[v].convex_set.vars
                )
            ]
            for u, v in tqdm(edges)
        ]

    def _create_edge_constraints(
        self, edges: List[Tuple[str, str]]
    ) -> List[List[Constraint]]:
        logger.info("Creating edge constraints...")
        constraints = []
        for u, v in tqdm(edges):
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
        logger.info("Generating edges...(parallel)")
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
        logger.info(f"{len(edges)} edges generated")
        return edges

    @staticmethod
    def _check_intersection(args):
        u_set, v_set = args
        return u_set.IntersectsWith(v_set)

    def _filter_edge_keys(self, edge_keys: List[Tuple[str, str]]):
        edges = []
        for u, v in edge_keys:
            if u in self.vertices and v in self.vertices:
                edges.append((u, v))
        return edges

    def _generate_contact_sets(
        self,
        contact_pair_modes: Dict[
            str, ContactPairMode
        ] = None,  # For loading a saved graph
        contact_set_mode_ids: List[Tuple] = None,  # For loading a saved graph
        vertex_exlcusion: List[str] = None,
        vertex_inclusion: List[str] = None,
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
        movable = obj_names + rob_names

        if contact_pair_modes is None or contact_set_mode_ids is None:
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
            body_pair_to_mode_names = {
                (body1, body2): [mode.id for mode in modes]
                for (body1, body2), modes in body_pair_to_modes.items()
            }
            mode_ids_to_mode = {
                mode.id: mode for modes in body_pair_to_modes.values() for mode in modes
            }
            self._contact_pair_modes = mode_ids_to_mode

            # Each set is the cartesian product of the modes for each object pair
            set_ids = list(product(*body_pair_to_mode_names.values()))
        else:
            logger.info(
                f"Loading {len(contact_pair_modes)} contact pair modes for {len(body_dict)} bodies..."
            )
            mode_ids_to_mode = contact_pair_modes
            self._contact_pair_modes = mode_ids_to_mode
            set_ids = contact_set_mode_ids

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

        logger.info(f"Generating contact sets for {len(set_ids)} sets...")

        all_contact_sets = [
            ContactSet(
                [mode_ids_to_mode[mode_id] for mode_id in set_id],
                set_force_constraints_dict[set_id],
                self.objects,
                self.robots,
            )
            for set_id in tqdm(set_ids)
        ]
        # Do not need to prune if loading from saved file, there shouldn't be empty sets
        if contact_set_mode_ids is None:
            logger.info(f"Pruning empty sets...")
            sets_to_keep = [
                contact_set
                for contact_set in tqdm(all_contact_sets)
                if not contact_set.set.IsEmpty()
            ]

            logger.info(
                f"{len(sets_to_keep)} sets remain after removing {len(all_contact_sets) - len(sets_to_keep)} empty sets"
            )
        else:
            sets_to_keep = all_contact_sets

        if vertex_exlcusion is not None:
            logger.info(f"Removing sets matching exclusion strings {vertex_exlcusion}")
            sets_to_keep = [
                contact_set
                for contact_set in tqdm(sets_to_keep)
                if not any(
                    vertex_exlcusion in str(contact_set.id)
                    for vertex_exlcusion in vertex_exlcusion
                )
            ]
            logger.info(f"{len(sets_to_keep)} sets remain after removing excluded sets")

        if vertex_inclusion is not None:
            logger.info(f"Filtering sets for inclusion strings {vertex_inclusion}")
            sets_to_keep = [
                contact_set
                for contact_set in tqdm(sets_to_keep)
                if any(
                    vertex_inclusion in str(contact_set.id)
                    for vertex_inclusion in vertex_inclusion
                )
            ]
            logger.info(
                f"{len(sets_to_keep)} sets remain after filtering for inclusion sets"
            )

        sets_to_keep_ids = [str(contact_set.id) for contact_set in sets_to_keep]
        return sets_to_keep, sets_to_keep_ids

    ### POST SOLVE ###

    def _post_solve(self, sol: ShortestPathSolution):
        """Post solve hook that is called after solving by the base graph class"""
        if sol.is_success:
            self.contact_spp_sol = self.create_contact_spp_sol(
                sol.vertex_path, sol.ambient_path
            )
        else:
            logger.warn("No Shortest Path Solution Found")

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

    ### PLOTTING AND ANIMATING ###

    def plot(self):
        for body in self.obstacles:
            body.plot()
        for body, pos in zip(self.objects, self.source_pos[: self.n_objects]):
            body.plot_at_position(
                pos=pos, label_vertices_faces=True, color=BodyColor["object"]
            )
        for body, pos in zip(self.robots, self.source_pos[self.n_objects :]):
            body.plot_at_position(
                pos=pos, label_vertices_faces=True, color=BodyColor["robot"]
            )
        for body, pos in zip(
            self.objects + self.robots,
            self.target_pos,
        ):
            body.plot_at_position(pos=pos, color=BodyColor["target"])

    def plot_samples_in_set(self, set_name: str, n_samples: int = 100, **kwargs):
        """Plots a single set"""
        options = {"facecolor": "mintcream", "edgecolor": "k", "zorder": 1}
        options.update(kwargs)
        plt.axis("equal")
        vertex = self.vertices[set_name]
        samples = vertex.convex_set.get_samples(n_samples)
        # logger.info(samples)
        raise NotImplementedError
        # Need to modify decompose_ambient_path
        obj_pos_trajectories, rob_pos_trajectories = self.decompose_ambient_path(
            samples
        )

        for j in range(len(samples)):
            # Plot object trajectories
            for i in range(obj_pos_trajectories.shape[0]):
                self.objects[i].plot_at_position(obj_pos_trajectories[i, j])
                # logger.info(f"obj_pos: {obj_pos_trajectories[i, j]}")
            for i in range(rob_pos_trajectories.shape[0]):
                self.robots[i].plot_at_position(rob_pos_trajectories[i, j])
                # logger.info(f"rob_pos: {rob_pos_trajectories[i, j]}")
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
        for obs in self.obstacles:
            obs.plot()
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
        import textwrap

        import matplotlib.animation as animation
        import matplotlib.patches as patches

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
            body.plot_at_position(self.target_pos[i], color=BodyColor["target"])

        label_text = [body.name for body in bodies]

        polygons = [
            patches.Polygon(
                body.geometry.vertices,
                color=BodyColor["object"]
                if body.mobility_type == MobilityType.UNACTUATED
                else BodyColor["robot"],
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
        # Wrap the text to fit within the width of the plot
        text_wrap_width = 80
        wrapped_text = "\n".join(
            textwrap.wrap(
                self.contact_spp_sol.vertex_path[transition_map[0]],
                width=text_wrap_width,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )

        # Adjust the position of the annotation
        vertex_annotation = ax.annotate(
            wrapped_text,
            xy=(0.5, 1),
            xycoords="axes fraction",
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round", fc="w"),
        )
        plt.subplots_adjust(
            top=0.85
        )  # Adjust subplot parameters to give the annotation enough space

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
                    wrapped_text = "\n".join(
                        textwrap.wrap(
                            self.contact_spp_sol.vertex_path[transition_map[i]],
                            width=text_wrap_width,
                            break_long_words=False,
                            break_on_hyphens=False,
                        )
                    )
                    vertex_annotation.set_text(wrapped_text)
                    # force_res_quivers[j].set_offsets(poly_offset[j] + trajs[i][j])
            return polygons

        anim = animation.FuncAnimation(
            fig, animate, frames=trajs.shape[0], interval=50, blit=True
        )
        return anim

    @staticmethod
    def _interpolate_positions(
        contact_sol: ContactShortestPathSolution, max_gap: float = 0.1
    ):
        # Input has shape (n_movable bodies, n_sets_in_path, n_pos_per_set, n_base_dim) OLD
        # Input has shape (n_pos, n_bodies, n_base_dim)
        trajs_in = contact_sol.pos_trajs
        # logger.info(f"trajs_in shape {trajs_in.shape}")
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

    ### SERIALIZATION METHODS ###

    def save_to_file(self, path: str):
        np.save(
            path,
            {
                "obs_params": [obs.params for obs in self.obstacles],
                "objs_params": [obj.params for obj in self.objects],
                "robs_params": [rob.params for rob in self.robots],
                "source_pos_objs": self.source_pos[: self.n_objects],
                "source_pos_robs": self.source_pos[self.n_objects :],
                "target_pos_objs": self.target_pos[: self.n_objects],
                "target_pos_robs": self.target_pos[self.n_objects :],
                "workspace": self.workspace,
                "vertex_exclusion": self.vertex_exclusion,
                "vertex_inclusion": self.vertex_inclusion,
                "contact_pair_mode_params": [
                    mode.params for mode in self._contact_pair_modes.values()
                ],
                "contact_set_mode_ids": [
                    tuple([mode.id for mode in v.convex_set.contact_pair_modes])
                    for v in self.vertices.values()
                    if isinstance(v.convex_set, ContactSet)
                ],
                "edge_keys": self.edge_keys,
            },
        )

    @classmethod
    def load_from_file(
        cls,
        path: str,
        vertex_inclusion: List[str] = None,
        vertex_exclusion: List[str] = None,
    ):
        data = np.load(path, allow_pickle=True).item()
        obs = [RigidBody.from_params(params) for params in data["obs_params"]]
        objs = [RigidBody.from_params(params) for params in data["objs_params"]]
        robs = [RigidBody.from_params(params) for params in data["robs_params"]]
        all_bodies = {body.name: body for body in obs + objs + robs}
        contact_pair_modes = {}
        for params in data["contact_pair_mode_params"]:
            body_a = all_bodies[params.body_a_name]
            body_b = all_bodies[params.body_b_name]
            mode = params.type(
                body_a,
                body_b,
                params.contact_location_a_type(body_a, params.contact_location_a_index),
                params.contact_location_b_type(body_b, params.contact_location_b_index),
            )
            contact_pair_modes[mode.id] = mode
        if vertex_inclusion is None:
            vertex_inclusion = data["vertex_inclusion"]
        if vertex_exclusion is None:
            vertex_exclusion = data["vertex_exclusion"]

        cg = cls(
            obs,
            objs,
            robs,
            data["source_pos_objs"],
            data["source_pos_robs"],
            data["target_pos_objs"],
            data["target_pos_robs"],
            data["workspace"],
            vertex_exclusion,
            vertex_inclusion,
            contact_pair_modes,
            data["contact_set_mode_ids"],
            data["edge_keys"],
        )
        return cg

    ### PROPERTIES ###

    @property
    def params(self):
        params = super().params
        params.source = self.source_pos
        params.target = self.target_pos
        return params

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
