import numpy as np

from large_gcs.graph.contact_cost_constraint_factory import (
    contact_vertex_cost_position_l1norm,
    contact_vertex_cost_position_l2norm,
)
from large_gcs.graph.incremental_contact_graph import IncrementalContactGraph
from large_gcs.graph_generators.contact_graph_generator import (
    ContactGraphGeneratorParams,
)


def test_vertex_cost_position():
    graph_file = ContactGraphGeneratorParams.inc_graph_file_path_from_name(
        "cg_stackpush_a2"
    )
    cg = IncrementalContactGraph.load_from_file(
        graph_file,
        should_incl_simul_mode_switches=False,
        should_add_const_edge_cost=True,
        should_add_gcs=True,
    )
    contact_pair_mode_ids = (
        "IC|obs0_f2-obj0_f0",
        "IC|obs0_f2-obj1_f0",
        "IC|obs0_f2-obj2_f0",
        "NC|obs0_f3-rob0_v1",
        "IC|obj0_f1-obj1_f3",
        "NC|obj0_f1-obj2_f3",
        "IC|obj0_f3-rob0_f1",
        "IC|obj1_f1-obj2_f3",
        "NC|obj1_f3-rob0_f1",
        "NC|obj2_f3-rob0_f1",
    )
    contact_set = cg._create_contact_set_from_contact_pair_mode_ids(
        contact_pair_mode_ids
    )
    vars = contact_set.vars

    # The forces do not satisfy the position/velocity constraints.
    point = np.array(
        [
            0,  # Variable('obj0_pos(0, 0)', Continuous),
            1,  # Variable('obj0_pos(0, 1)', Continuous),
            0,  # Variable('obj0_pos(1, 0)', Continuous),
            1,  # Variable('obj0_pos(1, 1)', Continuous),
            0,  # Variable('obj1_pos(0, 0)', Continuous),
            1,  # Variable('obj1_pos(0, 1)', Continuous),
            0,  # Variable('obj1_pos(1, 0)', Continuous),
            1,  # Variable('obj1_pos(1, 1)', Continuous),
            0,  # Variable('obj2_pos(0, 0)', Continuous),
            1,  # Variable('obj2_pos(0, 1)', Continuous),
            0,  # Variable('obj2_pos(1, 0)', Continuous),
            1,  # Variable('obj2_pos(1, 1)', Continuous),
            5,  # Variable('rob0_pos(0, 0)', Continuous),
            7,  # Variable('rob0_pos(0, 1)', Continuous),
            5,  # Variable('rob0_pos(1, 0)', Continuous),
            7,  # Variable('rob0_pos(1, 1)', Continuous),
            11,  # Variable('rob0_force_act(0)', Continuous),
            11,  # Variable('rob0_force_act(1)', Continuous),
            11,  # Variable('IC|obs0_f2-obj0_f0_force_mag_AB', Continuous),
            11,  # Variable('IC|obs0_f2-obj1_f0_force_mag_AB', Continuous),
            11,  # Variable('IC|obs0_f2-obj2_f0_force_mag_AB', Continuous),
            11,  # Variable('IC|obj0_f1-obj1_f3_force_mag_AB', Continuous),
            11,  # Variable('IC|obj0_f3-rob0_f1_force_mag_AB', Continuous),
            11,  # Variable('IC|obj1_f1-obj2_f3_force_mag_AB', Continuous)
        ],
    )

    # Test L2 norm
    a = np.hstack((np.ones(vars.n_objects * 2), [2, 2]))
    no_scaling_ans = np.linalg.norm(a)
    cost = contact_vertex_cost_position_l2norm(contact_set.vars, scaling=1)
    assert cost.Eval(point)[0] == no_scaling_ans

    SCALING = 11
    cost = contact_vertex_cost_position_l2norm(contact_set.vars, scaling=SCALING)
    assert cost.Eval(point)[0] == no_scaling_ans * SCALING

    # Test L1 norm
    no_scaling_ans = np.linalg.norm(a, ord=1)
    cost = contact_vertex_cost_position_l1norm(contact_set.vars, scaling=1)
    assert cost.Eval(point)[0] == no_scaling_ans

    cost = contact_vertex_cost_position_l1norm(contact_set.vars, scaling=SCALING)
    assert cost.Eval(point)[0] == no_scaling_ans * SCALING

    # from IPython.display import Markdown, display
    # display(Markdown("$$" + cost.ToLatex(contact_set.vars.all) + "$$"))
