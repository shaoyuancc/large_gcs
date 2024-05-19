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
    x_buffer: Optional[np.ndarray] = None,
    y_buffer: Optional[np.ndarray] = None,
    filepath: Optional[Path] = None,
    target_regions: Optional[List[Polyhedron]] = None,
    add_legend: bool = False,
    use_type_1_font: bool = True,
    keyframe_idxs: Optional[List[int]] = None,
    use_paper_params: bool = True,  # TODO(bernhardpg): Set to false
):

    if x_buffer is None:
        x_buffer = np.array([1.4, 1.4])

    if y_buffer is None:
        y_buffer = np.array([1.4, 1.4])

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
        num_keyframes = int(np.ceil(n_steps / 15))

    if use_paper_params:
        # NOTE: These are specific parameters that we use to get the
        # figures we want in the paper, and should be removed for
        # general use.
        # They are made to match the trajs generated from
        # WAFR_experiments/trajectory_figures.yaml
        if num_keyframes == 9:  # cg_maze_b1
            num_keyframes = 6
            keyframe_idxs = [0, 32, 50, 72, 86, 119]
            keyframe_idxs.append(n_steps)
            x_buffer = np.array([0.8, 0.8])
            y_buffer = np.array([1.0, 1.0])

        elif num_keyframes == 7:  # STACK
            num_keyframes = 6
            # keyframe_idxs = [0, 32, 59, 70, 80, 95]
            keyframe_idxs = [0, 16, 22, 35, 66, 81]
            keyframe_idxs.append(n_steps)
            x_buffer = np.array([0.8, 0.8])
            y_buffer = np.array([1.0, 1.0])

            add_legend = True
            legend_loc = "lower left"

        elif num_keyframes == 4:  # cg_trichal4
            # Adjust these numbers to adjust what frames the keyframes start at:
            keyframe_idxs = [0, 14, 28, 46]

            # this step is needed for downstream code
            keyframe_idxs.append(n_steps)
            y_buffer = np.array([1.2, 1.2])
            x_buffer = np.array([1.5, 1.5])

            add_legend = True
            legend_loc = "upper left"

    ROBOT_COLOR = DARKSEAGREEN2.diffuse()
    OBSTACLE_COLOR = AZURE3.diffuse()
    OBJECT_COLOR = BISQUE3.diffuse()

    EDGE_COLOR = BLACK.diffuse()

    EMERALDGREEN.diffuse()

    START_TRANSPARENCY = 0.3
    END_TRANSPARENCY = 1.0

    fig_height = 4
    if add_legend:
        fig_height = 5

    subplot_width = 4

    fig, axs = plt.subplots(
        1, num_keyframes, figsize=(subplot_width * num_keyframes, fig_height)
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
            x_min -= x_buffer[0]
            x_max += x_buffer[1]

        if y_buffer is not None:
            y_min -= y_buffer[0]
            y_max += y_buffer[1]

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Hide the axes, including the spines, ticks, labels, and title
        ax.set_axis_off()

    n_objects = len(objects)
    n_robots = len(robots)

    # Plot goal regions
    if target_regions is not None:
        goal_kwargs = {
            "edgecolor": "none",
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
        # Add one here so we display the last and first frames in a keyframe twice (once per frame)
        steps_per_axs = [
            list(range(idx_curr, idx_next + 1))
            for idx_curr, idx_next in zip(keyframe_idxs[:-1], keyframe_idxs[1:])
        ]
        # Remove the last one
        steps_per_axs[-1].pop()
    else:
        steps_per_axs = split_numbers_into_sublists(n_steps, len(axs))

    transparencies = [
        np.linspace(START_TRANSPARENCY, END_TRANSPARENCY, len(steps)).tolist()
        # Alternatively, use a logscale:
        # np.logspace(
        #     np.log10(START_TRANSPARENCY), np.log10(END_TRANSPARENCY), num=len(steps)
        # )
        for steps in steps_per_axs
    ]
    for ax, alphas, steps in zip(axs, transparencies, steps_per_axs):
        for alpha, step_idx in zip(alphas, steps):
            for obj_idx in range(n_objects):
                objects[obj_idx].plot_at_position(
                    pos_trajs[step_idx, obj_idx],
                    facecolor=OBJECT_COLOR,
                    label_body=False,
                    label_vertices_faces=False,
                    edgecolor=EDGE_COLOR,
                    ax=ax,
                    alpha=alpha,
                )
            for obj_idx in range(n_robots):
                robots[obj_idx].plot_at_position(
                    pos_trajs[step_idx, obj_idx + n_objects],
                    label_body=False,
                    facecolor=ROBOT_COLOR,
                    label_vertices_faces=False,
                    edgecolor=EDGE_COLOR,
                    ax=ax,
                    alpha=alpha,
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
        fig.legend(
            handles=custom_patches,
            handlelength=2.5,
            fontsize=28,
            ncol=2,
            loc=legend_loc,  # type: ignore
        )

    # Adjust layout to make room for the legend
    if add_legend:
        if legend_loc == "upper left":
            fig.tight_layout(rect=(0, 0, 1, 0.7))
        elif legend_loc == "lower left":
            fig.tight_layout(rect=(0, 0.3, 1, 1.0))
    else:
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
