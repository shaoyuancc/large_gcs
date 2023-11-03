from typing import List

from large_gcs.abstraction_models.abstraction_model import (
    AbstractionModel,
    AbstractionModelGenerator,
)
from large_gcs.abstraction_models.gcshastar_node import (
    ContextNode,
    GCSHANode,
    StatementNode,
)
from large_gcs.cost_estimators.factored_collision_free_ce import FactoredCollisionFreeCE
from large_gcs.graph.contact_graph import ContactGraph
from large_gcs.graph.factored_collision_free_graph import FactoredCollisionFreeGraph
from large_gcs.graph.incremental_relaxed_contact_graph import (
    IncrementalRelaxedContactGraph,
)


class ContactThree(AbstractionModelGenerator):
    def generate(self, concrete_graph: ContactGraph) -> AbstractionModel:
        """
        Generate the abstraction model for the contact two abstraction.
        """
        # Define level 0 graph
        graphs = []
        graphs.append(concrete_graph)
        # Define level 1 graph
        cg_relaxed = IncrementalRelaxedContactGraph.from_inc_contact_graph(
            concrete_graph
        )
        graphs.append(cg_relaxed)
        # Define level 2 graph
        # Right now the body is hardcoded and limited to one, soon we will extend this to multiple bodies.
        body = graphs[0].objects[0]
        cg_factored_cfree = FactoredCollisionFreeGraph(
            movable_body=body,
            static_obstacles=graphs[0].obstacles,
            source_pos=graphs[0].source_pos[0],
            # target_pos=graphs[0].target_pos[0],
            target_region_params=graphs[0].target_region_params[0],
            cost_scaling=1.0,
            workspace=graphs[0].workspace,
            add_source_set=True,
        )
        graphs.append(cg_factored_cfree)

        def abs_full_problem(n: StatementNode) -> List[StatementNode]:
            new_abs_level = n.abs_level + 1
            abstract_nodes = []
            # HACK find a better way to handle this case/feed this info in.
            if n.vertex_name == graphs[n.abs_level].target_name:
                relaxed_name = [graphs[new_abs_level].target_name]
            else:
                relaxed_name = (
                    IncrementalRelaxedContactGraph.full_to_relaxed_contact_vertex_name(
                        n.vertex_name
                    )
                )

            abstract_nodes.append(
                StatementNode(
                    priority=None,
                    abs_level=new_abs_level,
                    vertex_name=relaxed_name,
                    weight=None,
                    path=[],
                )
            )
            return abstract_nodes

        def abs_relaxed_contact(n: StatementNode) -> List[StatementNode]:
            new_abs_level = n.abs_level + 1
            abstract_nodes = []
            # HACK find a better way to handle this case/feed this info in.
            if n.vertex_name == graphs[n.abs_level].target_name:
                split_names = [graphs[new_abs_level].target_name]
            else:
                split_names = FactoredCollisionFreeCE.convert_to_cfree_vertex_names(
                    n.vertex_name
                )
            for name in split_names:
                # Temporary HACK, filter out robot sets (and we know they are all grouped together after object sets)
                if "rob" in name:
                    break
                abstract_nodes.append(
                    StatementNode(
                        priority=None,
                        abs_level=new_abs_level,
                        vertex_name=name,
                        weight=None,
                        path=[],
                    )
                )
            return abstract_nodes

        def abs_factored_cfree_bodies_with_goals(
            n: StatementNode,
        ) -> List[StatementNode]:
            new_abs_level = n.abs_level + 1
            new_vertex_name = "START"
            return [
                StatementNode(
                    priority=None,
                    abs_level=new_abs_level,
                    vertex_name=new_vertex_name,
                    weight=0,
                    path=[],
                )
            ]

        abs_fns = [
            abs_full_problem,
            abs_relaxed_contact,
            abs_factored_cfree_bodies_with_goals,
        ]

        return AbstractionModel(graphs=graphs, abs_fns=abs_fns)
