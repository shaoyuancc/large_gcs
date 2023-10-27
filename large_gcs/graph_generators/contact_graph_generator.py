import os
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from large_gcs.contact.contact_regions_set import ContactRegionParams
from large_gcs.contact.rigid_body import BodyColor, MobilityType, RigidBody
from large_gcs.geometry.polyhedron import Polyhedron
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.incremental_contact_graph import IncrementalContactGraph


@dataclass
class ContactGraphGeneratorParams:
    name: str
    obs_vertices: List
    obj_vertices: List
    rob_vertices: List
    source_obj_pos: List
    source_rob_pos: List
    n_pos_per_set: int
    workspace: List
    target_obj_pos: List = None
    target_rob_pos: List = None
    target_region_params: List[ContactRegionParams] = None

    def __post_init__(self):
        self.obs_vertices = np.array(self.obs_vertices)
        self.obj_vertices = np.array(self.obj_vertices)
        self.rob_vertices = np.array(self.rob_vertices)
        self.source_obj_pos = np.array(self.source_obj_pos)
        self.source_rob_pos = np.array(self.source_rob_pos)

        self.workspace = np.array(self.workspace)

        body_dims = set()
        if self.obs_vertices.size > 0:
            body_dims.add(self.obs_vertices.shape[2])
        if self.obj_vertices.size > 0:
            body_dims.add(self.obj_vertices.shape[2])
        if self.rob_vertices.size > 0:
            body_dims.add(self.rob_vertices.shape[2])
        assert len(body_dims) == 1, "All bodies must have same dimension"
        n_dim = body_dims.pop()
        if self.target_obj_pos is not None:
            self.target_obj_pos = np.array(self.target_obj_pos)
            assert (
                self.source_obj_pos.shape
                == self.target_obj_pos.shape
                == (self.obj_vertices.shape[0], n_dim)
            )
            self.target_obj_pos = list(self.target_obj_pos)
        if self.target_rob_pos is not None:
            self.target_rob_pos = np.array(self.target_rob_pos)
            assert (
                self.source_rob_pos.shape
                == self.target_rob_pos.shape
                == (self.rob_vertices.shape[0], n_dim)
            )
            self.target_rob_pos = list(self.target_rob_pos)
        assert self.n_pos_per_set > 1, "Need at least 2 positions per set"
        assert self.workspace.shape == (n_dim, 2)

        self.source_obj_pos = list(self.source_obj_pos)
        self.source_rob_pos = list(self.source_rob_pos)

    @property
    def graph_file_path(self) -> str:
        return self.graph_file_path_from_name(self.name)

    @property
    def inc_graph_file_path(self) -> str:
        return self.graph_file_path_from_name(self.name + "_inc")

    @staticmethod
    def graph_file_path_from_name(name: str) -> str:
        return os.path.join(os.environ["PROJECT_ROOT"], "example_graphs", name + ".npy")

    @staticmethod
    def inc_graph_file_path_from_name(name: str) -> str:
        return os.path.join(
            os.environ["PROJECT_ROOT"], "example_graphs", name + "_inc.npy"
        )


class ContactGraphGenerator:
    def __init__(self, params: ContactGraphGeneratorParams):
        self._params = params
        self._obs = []
        self._objs = []
        self._robs = []
        for i in range(len(self._params.obs_vertices)):
            self._obs.append(
                RigidBody(
                    name=f"obs{i}",
                    geometry=Polyhedron.from_vertices(self._params.obs_vertices[i]),
                    mobility_type=MobilityType.STATIC,
                    n_pos_points=self._params.n_pos_per_set,
                )
            )
        for i in range(len(self._params.obj_vertices)):
            self._objs.append(
                RigidBody(
                    name=f"obj{i}",
                    geometry=Polyhedron.from_vertices(self._params.obj_vertices[i]),
                    mobility_type=MobilityType.UNACTUATED,
                    n_pos_points=self._params.n_pos_per_set,
                )
            )
        for i in range(len(self._params.rob_vertices)):
            self._robs.append(
                RigidBody(
                    name=f"rob{i}",
                    geometry=Polyhedron.from_vertices(self._params.rob_vertices[i]),
                    mobility_type=MobilityType.ACTUATED,
                    n_pos_points=self._params.n_pos_per_set,
                )
            )

    def generate(self, save_to_file=True) -> ContactGraph:
        contact_graph = ContactGraph(
            static_obstacles=self._obs,
            unactuated_objects=self._objs,
            actuated_robots=self._robs,
            source_pos_objs=self._params.source_obj_pos,
            source_pos_robs=self._params.source_rob_pos,
            target_pos_objs=self._params.target_obj_pos,
            target_pos_robs=self._params.target_rob_pos,
            target_region_params=self._params.target_region_params,
            workspace=self._params.workspace,
            vertex_exclusion=None,
            vertex_inclusion=None,
        )
        if save_to_file:
            contact_graph.save_to_file(self._params.graph_file_path)

        return contact_graph

    def generate_incremental_contact_graph(
        self, save_to_file=True
    ) -> IncrementalContactGraph:
        contact_graph = IncrementalContactGraph(
            static_obstacles=self._obs,
            unactuated_objects=self._objs,
            actuated_robots=self._robs,
            source_pos_objs=self._params.source_obj_pos,
            source_pos_robs=self._params.source_rob_pos,
            target_pos_objs=self._params.target_obj_pos,
            target_pos_robs=self._params.target_rob_pos,
            target_region_params=self._params.target_region_params,
            workspace=self._params.workspace,
            should_add_gcs=True,
        )
        if save_to_file:
            contact_graph.save_to_file(self._params.inc_graph_file_path)

        return contact_graph

    def plot(self):
        if self._params.workspace is not None:
            plt.axes(xlim=self._params.workspace[0], ylim=self._params.workspace[1])
        plt.gca().set_aspect("equal")

        for body in self._obs:
            body.plot()
        for body, pos in zip(self._objs, self._params.source_obj_pos):
            body.plot_at_position(
                pos=pos, label_vertices_faces=True, color=BodyColor["object"]
            )
        for body, pos in zip(self._robs, self._params.source_rob_pos):
            body.plot_at_position(
                pos=pos, label_vertices_faces=True, color=BodyColor["robot"]
            )
        if self._params.target_region_params is not None:
            for params in self._params.target_region_params:
                region = Polyhedron.from_vertices(params.region_vertices)
                region.plot(color=BodyColor["target"], alpha=0.3)
        elif (
            self._params.target_obj_pos is not None
            and self._params.target_rob_pos is not None
        ):
            for body, pos in zip(
                self._objs + self._robs,
                self._params.target_obj_pos + self._params.target_rob_pos,
            ):
                body.plot_at_position(pos=pos, color=BodyColor["target"])
