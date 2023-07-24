import os
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from large_gcs.contact.rigid_body import BodyColor, MobilityType, RigidBody
from large_gcs.geometry.polyhedron import Polyhedron
from large_gcs.graph.contact_graph import ContactGraph


@dataclass
class ContactGraphGeneratorParams:
    name: str
    obs_vertices: List
    obj_vertices: List
    rob_vertices: List
    source_obj_pos: List
    source_rob_pos: List
    target_obj_pos: List
    target_rob_pos: List
    n_pos_per_set: int
    workspace: List

    def __post_init__(self):
        self.obs_vertices = np.array(self.obs_vertices)
        self.obj_vertices = np.array(self.obj_vertices)
        self.rob_vertices = np.array(self.rob_vertices)
        self.source_obj_pos = np.array(self.source_obj_pos)
        self.source_rob_pos = np.array(self.source_rob_pos)
        self.target_obj_pos = np.array(self.target_obj_pos)
        self.target_rob_pos = np.array(self.target_rob_pos)
        self.workspace = np.array(self.workspace)

        if self.obs_vertices.size > 0:
            assert (
                self.obs_vertices.shape[2] == self.rob_vertices.shape[2]
            ), "All bodies must have same dimension"
        if self.obj_vertices.size > 0:
            assert (
                self.obj_vertices.shape[2] == self.rob_vertices.shape[2]
            ), "All bodies must have same dimension"
        n_dim = self.rob_vertices.shape[2]
        assert (
            self.source_obj_pos.shape
            == self.target_obj_pos.shape
            == (self.obj_vertices.shape[0], n_dim)
        )
        assert (
            self.source_rob_pos.shape
            == self.target_rob_pos.shape
            == (self.rob_vertices.shape[0], n_dim)
        )
        assert self.n_pos_per_set > 1, "Need at least 2 positions per set"
        assert self.workspace.shape == (n_dim, 2)

        self.source_obj_pos = list(self.source_obj_pos)
        self.source_rob_pos = list(self.source_rob_pos)
        self.target_obj_pos = list(self.target_obj_pos)
        self.target_rob_pos = list(self.target_rob_pos)

    @property
    def graph_file_path(self):
        return self.graph_file_path_from_name(self.name)

    @staticmethod
    def graph_file_path_from_name(name: str):
        return os.path.join(os.environ["PROJECT_ROOT"], "example_graphs", name + ".npy")


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

    def generate(self) -> ContactGraph:
        contact_graph = ContactGraph(
            self._obs,
            self._objs,
            self._robs,
            self._params.source_obj_pos,
            self._params.source_rob_pos,
            self._params.target_obj_pos,
            self._params.target_rob_pos,
            workspace=self._params.workspace,
            vertex_exclusion=None,
            vertex_inclusion=None,
        )

        contact_graph.save_to_file(self._params.graph_file_path)

    def plot(self):
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
        for body, pos in zip(
            self._objs + self._robs,
            np.concatenate((self._params.target_obj_pos, self._params.target_rob_pos)),
        ):
            body.plot_at_position(pos=pos, color=BodyColor["target"])
