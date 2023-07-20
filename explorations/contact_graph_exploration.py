from large_gcs.contact.contact_location import *
from large_gcs.contact.contact_pair_mode import *
from large_gcs.contact.rigid_body import *
from large_gcs.geometry.polyhedron import Polyhedron
from large_gcs.graph.contact_graph import ContactGraph
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    # Rest of your code

    # 3 items triangle challenge
    obs_vertices = [[[-1, 2], [-1, -1], [2, 2]]]
    obj_vertices = [[[1, 0.5], [1, -0.5], [2, -0.5], [2, 0.5]]]
    rob_vertices = [[[3, 1.5], [3, 0], [3.5, 0]]]
    source_obj_pos = [[1.5, 0.5]]
    source_rob_pos = [[3.25, 0]]
    target_obj_pos = [[-1.5, 0]]
    target_rob_pos = [[-3, 0]]

    # # 2 movable items
    # obs_vertices = []
    # obj_vertices = [np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) + np.array([2.5, 0.5])]
    # rob_vertices = [np.array([[-1, -1], [-1.5, -0.5], [-1.2, -1.5]])]
    # source_obj_pos = [[0, 0]]
    # source_rob_pos = [[-2, -2]]
    # target_obj_pos = [[2, 0]]
    # target_rob_pos = [[2.5, 2]]

    obs = []
    objs = []
    robs = []
    n_pos_per_set = 2
    for i in range(len(obs_vertices)):
        obs.append(
            RigidBody(
                name=f"obs{i}",
                geometry=Polyhedron.from_vertices(obs_vertices[i]),
                mobility_type=MobilityType.STATIC,
                n_pos_points=n_pos_per_set,
            )
        )
    for i in range(len(obj_vertices)):
        objs.append(
            RigidBody(
                name=f"obj{i}",
                geometry=Polyhedron.from_vertices(obj_vertices[i]),
                mobility_type=MobilityType.UNACTUATED,
                n_pos_points=n_pos_per_set,
            )
        )
    for i in range(len(rob_vertices)):
        robs.append(
            RigidBody(
                name=f"rob{i}",
                geometry=Polyhedron.from_vertices(rob_vertices[i]),
                mobility_type=MobilityType.ACTUATED,
                n_pos_points=n_pos_per_set,
            )
        )
    all_rigid_bodies = obs + objs + robs

    contact_graph = ContactGraph(
        obs,
        objs,
        robs,
        source_obj_pos,
        source_rob_pos,
        target_obj_pos,
        target_rob_pos,
        workspace=[[-3.5, 3.5], [-2.5, 2.5]],
    )

    print(contact_graph.params)

    # time the solve
    start_time = time.time()
    print("Solve started at ", time.ctime(start_time))
    sol = contact_graph.solve(use_convex_relaxation=False)
    end_time = time.time()
    print("Solve ended at ", time.ctime(end_time))
    print("Solve time: ", end_time - start_time)
    vertex_names, ambient_path = zip(*sol.path)
    print(vertex_names)
    print(f"sol time: {sol.time} s")
    contact_sol = contact_graph.contact_spp_sol
    anim = contact_graph.animate_solution()
    # save as mp4
    anim.save("contact_graph_demo_triangle_challenge.mp4")
