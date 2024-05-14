from pathlib import Path
from typing import List, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from large_gcs.contact.rigid_body import RigidBody


def plot_trajectory(
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
