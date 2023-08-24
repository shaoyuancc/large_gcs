from large_gcs.cost_estimators.factored_collision_free_ce import FactoredCollisionFreeCE


def test_convert_to_cfree_vertex_names_mult_objs():
    vertex_name = "('NC|obs0_f3-obj0_v1', 'NC|obs0_f3-obj1_v1', 'NC|obs0_f3-rob0_v1', 'NC|obj0_f1-obj1_f3', 'NC|obj0_f2-rob0_f0', 'NC|obj1_f2-rob0_f0')"
    cfree_vertex_names = [
        "('NC|obs0_f3-obj0_v1',)",
        "('NC|obs0_f3-obj1_v1',)",
        "('NC|obs0_f3-rob0_v1',)",
    ]
    assert (
        FactoredCollisionFreeCE._convert_to_cfree_vertex_names(vertex_name)
        == cfree_vertex_names
    )
