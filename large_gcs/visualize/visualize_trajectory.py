from pathlib import Path
from typing import List, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from large_gcs.contact.rigid_body import RigidBody
from large_gcs.geometry.polyhedron import Polyhedron
from large_gcs.utils.utils import split_numbers_into_sublists
from large_gcs.visualize.colors import (
    AZURE3,
    BISQUE3,
    BLACK,
    CRIMSON,
    DARKSEAGREEN2,
    DARKSLATEGRAY1,
    EMERALDGREEN,
    FIREBRICK3,
)


def plot_trajectory(
    pos_trajs: np.ndarray,  # (n_steps, n_objects + n_robots)
    obstacles: List[RigidBody],
    objects: List[RigidBody],
    robots: List[RigidBody],
    workspace: np.ndarray,  # (2, 2)
    filepath: Optional[Path] = None,
    target_pos: Optional[List[np.ndarray]] = None,
    target_regions: Optional[List[Polyhedron]] = None,
):
    num_keyframes = 5
    fig_height = 4

    ROBOT_COLOR = DARKSEAGREEN2.diffuse()
    OBSTACLE_COLOR = AZURE3.diffuse()
    OBJECT_COLOR = BISQUE3.diffuse()

    EDGE_COLOR = BLACK.diffuse()

    START_COLOR = CRIMSON.diffuse()
    GOAL_COLOR = EMERALDGREEN.diffuse()

    START_TRANSPARENCY = 0.0
    END_TRANSPARENCY = 1.0

    fig, axs = plt.subplots(
        1, num_keyframes, figsize=(fig_height * num_keyframes, fig_height)
    )

    for ax in axs:
        ax.set_aspect("equal")

        x_min, x_max = workspace[0]
        ax.set_xlim(x_min, x_max)

        y_min, y_max = workspace[1]
        ax.set_ylim(y_min, y_max)

        # Hide the axes, including the spines, ticks, labels, and title
        ax.set_axis_off()

    n_objects = len(objects)
    n_robots = len(robots)

    # Plot goal positions
    if target_pos is not None:
        raise RuntimeError(
            "This part of the function has not been tested\
            (if it works, feel free to remove this warning.)"
        )
        for i, body in enumerate(bodies):
            body.plot_at_position(self.target_pos[i], color=BodyColor["target"])
    elif target_regions is not None:
        for region in target_regions:
            region.plot(color=GOAL_COLOR, alpha=0.2)

    for ax in axs:
        for obs in obstacles:
            obs.plot_at_com(
                facecolor=OBSTACLE_COLOR,
                label_body=False,
                label_vertices_faces=False,
                edgecolor=EDGE_COLOR,
                ax=ax,
            )

    n_steps = pos_trajs.shape[0]

    steps_per_axs = split_numbers_into_sublists(n_steps, len(axs))
    transparencies = np.concatenate(
        [
            np.linspace(START_TRANSPARENCY, END_TRANSPARENCY, len(steps))
            for steps in steps_per_axs
        ]
    )
    for ax, steps in zip(axs, steps_per_axs):

        for step_idx in steps:
            for obj_idx in range(n_objects):
                objects[obj_idx].plot_at_position(
                    pos_trajs[step_idx, obj_idx],
                    facecolor=OBJECT_COLOR,
                    label_body=False,
                    label_vertices_faces=False,
                    edgecolor=EDGE_COLOR,
                    ax=ax,
                    alpha=transparencies[step_idx],
                )
            for obj_idx in range(n_robots):
                robots[obj_idx].plot_at_position(
                    pos_trajs[step_idx, obj_idx + n_objects],
                    label_body=False,
                    facecolor=ROBOT_COLOR,
                    label_vertices_faces=False,
                    edgecolor=EDGE_COLOR,
                    ax=ax,
                    alpha=transparencies[step_idx],
                )

    if filepath:
        fig.savefig(filepath, format="pdf")
        plt.close()
    else:
        plt.show()


# This is the original plotting function that was originally implemented
def plot_trajectory_legacy(
    pos_trajs: np.ndarray,  # (n_steps, n_objects + n_robots)
    obstacles: List[RigidBody],
    objects: List[RigidBody],
    robots: List[RigidBody],
    workspace: np.ndarray,  # (2, 2)
    filepath: Optional[Path] = None,
):
    fig = plt.figure()
    ax = plt.axes(xlim=workspace[0], ylim=workspace[1])
    ax.set_aspect("equal")

    n_objects = len(objects)
    n_robots = len(robots)

    for obs in obstacles:
        obs.plot()
    n_steps = pos_trajs.shape[0]
    for i in range(n_steps):
        for j in range(n_objects):
            objects[j].plot_at_position(
                pos_trajs[i, j],
                facecolor="none",
                label_body=False,
                edgecolor=cm.rainbow(i / n_steps),  # type: ignore
            )
        for j in range(n_robots):
            robots[j].plot_at_position(
                pos_trajs[i, j + n_objects],
                label_body=False,
                facecolor="none",
                edgecolor=cm.rainbow(i / n_steps),  # type: ignore
            )

    # Add a color bar
    sm = plt.cm.ScalarMappable(
        cmap=cm.rainbow, norm=plt.Normalize(vmin=0, vmax=n_steps)  # type: ignore
    )

    plt.colorbar(sm, ax=ax)
    plt.grid()

    if filepath:
        fig.savefig(filepath, format="pdf")
        plt.close()
    else:
        plt.show()
