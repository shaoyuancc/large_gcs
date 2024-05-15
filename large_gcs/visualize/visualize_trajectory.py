from pathlib import Path
from typing import List, Optional

import matplotlib.cm as cm
import matplotlib.patches as mpatches
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
    WHITE,
)


def plot_trajectory(
    pos_trajs: np.ndarray,  # (n_steps, n_objects + n_robots)
    obstacles: List[RigidBody],
    objects: List[RigidBody],
    robots: List[RigidBody],
    workspace: Optional[np.ndarray] = None,  # (2, 2)
    x_buffer: Optional[float] = 1.7,
    y_buffer: Optional[float] = 1.7,
    filepath: Optional[Path] = None,
    target_regions: Optional[List[Polyhedron]] = None,
    add_legend: bool = False,
    use_type_1_font: bool = True,
    keyframe_idxs: Optional[List[int]] = None,
):
    if use_type_1_font:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["ps.useafm"] = True
        plt.rcParams["pdf.use14corefonts"] = True
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.serif"] = "Computer Modern Roman"

    n_steps = pos_trajs.shape[0]

    if keyframe_idxs is not None:
        if max(keyframe_idxs) >= n_steps:
            raise RuntimeError("Last keyframe is after end of trajectory.")

        num_keyframes = len(keyframe_idxs)

        # Make sure we plot until the end
        keyframe_idxs.append(n_steps)
    else:
        num_keyframes = int(np.ceil(n_steps / 30))

    fig_height = 4

    ROBOT_COLOR = DARKSEAGREEN2.diffuse()
    OBSTACLE_COLOR = AZURE3.diffuse()
    OBJECT_COLOR = BISQUE3.diffuse()

    EDGE_COLOR = BLACK.diffuse()

    GOAL_COLOR = EMERALDGREEN.diffuse()

    START_TRANSPARENCY = 0.3
    END_TRANSPARENCY = 1.0

    fig, axs = plt.subplots(
        1, num_keyframes, figsize=(fig_height * num_keyframes, fig_height)
    )

    # ensure we can still iterate through the "axes" even if we only have one
    if num_keyframes == 1:
        axs = [axs]

    for ax in axs:
        ax.set_aspect("equal")

        if workspace is not None:
            x_min, x_max = workspace[0]
            y_min, y_max = workspace[1]
        else:
            x_min = np.min(pos_trajs[:, :, 0])
            x_max = np.max(pos_trajs[:, :, 0])
            y_min = np.min(pos_trajs[:, :, 1])
            y_max = np.max(pos_trajs[:, :, 1])

        if x_buffer is not None:
            x_min -= x_buffer
            x_max += x_buffer

        if y_buffer is not None:
            y_min -= y_buffer
            y_max += y_buffer

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Hide the axes, including the spines, ticks, labels, and title
        ax.set_axis_off()

    n_objects = len(objects)
    n_robots = len(robots)

    # Plot goal regions
    if target_regions is not None:
        goal_kwargs = {
            "edgecolor": BLACK,
            "facecolor": "none",
            "hatch": "....",
            "linewidth": 1,
            "alpha": 0.3,
        }
        for ax in axs:
            for region in target_regions:
                region.plot(**goal_kwargs, ax=ax)

    for ax in axs:
        for obs in obstacles:
            obs.plot_at_com(
                facecolor=OBSTACLE_COLOR,
                label_body=False,
                label_vertices_faces=False,
                edgecolor=EDGE_COLOR,
                ax=ax,
            )

    if keyframe_idxs:
        steps_per_axs = [
            list(range(idx_curr, idx_next))
            for idx_curr, idx_next in zip(keyframe_idxs[:-1], keyframe_idxs[1:])
        ]
    else:
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

    if add_legend:
        # Create a list of patches to use as legend handles
        custom_patches = [
            mpatches.Patch(color=color, label=label)
            for label, color in zip(
                ["Static obstacles", "Unactuated object", "Actuated robot"],
                [OBSTACLE_COLOR, OBJECT_COLOR, ROBOT_COLOR],
            )
        ]
        if target_regions is not None:
            goal_patch = mpatches.Patch(
                **goal_kwargs, label="Object target region"
            )  # type: ignore
            custom_patches += [goal_patch]

        # Creating the custom legend
        axs[0].legend(
            handles=custom_patches,
            handlelength=2.5,
            fontsize=12,
            # loc="lower left",
        )

    fig.tight_layout()

    if filepath:
        fig.savefig(filepath)
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
