import logging
from typing import List

from large_gcs.algorithms.search_algorithm import AlgMetrics, SearchNode
from large_gcs.domination_checkers.ah_containment_domination_checker import (
    AHContainmentDominationChecker,
)
from large_gcs.domination_checkers.ah_containment_last_pos import (
    ReachesCheaperLastPosContainment,
    ReachesNewLastPosContainment,
)
from large_gcs.domination_checkers.domination_checker import DominationChecker
from large_gcs.domination_checkers.sampling_domination_checker import (
    SamplingDominationChecker,
)
from large_gcs.domination_checkers.sampling_last_pos import (
    ReachesCheaperLastPosSampling,
    ReachesNewLastPosSampling,
)
from wandb import Graph

logger = logging.getLogger(__name__)


class SamplingContainmentDominationChecker(DominationChecker):
    @property
    def sampling_domination_checker(self) -> SamplingDominationChecker:
        return self._sampling_domination_checker

    @property
    def containment_domination_checker(self) -> AHContainmentDominationChecker:
        return self._containment_domination_checker

    def is_dominated(
        self, candidate_node: SearchNode, alternate_nodes: List[SearchNode]
    ) -> bool:

        if self.sampling_domination_checker.is_dominated(
            candidate_node, alternate_nodes
        ) and self.containment_domination_checker.is_dominated(
            candidate_node, alternate_nodes
        ):
            return True
        return False

    def set_alg_metrics(self, alg_metrics: AlgMetrics):
        self.sampling_domination_checker.set_alg_metrics(alg_metrics)
        self.containment_domination_checker.set_alg_metrics(alg_metrics)


class ReachesNewLastPosSamplingContainment(SamplingContainmentDominationChecker):
    def __init__(
        self,
        graph: Graph,
        num_samples_per_vertex: int,
        should_use_candidate_sol: bool = False,
        containment_condition: int = -1,
    ):
        super().__init__(graph=graph)
        self._sampling_domination_checker = ReachesNewLastPosSampling(
            graph=graph,
            num_samples_per_vertex=num_samples_per_vertex,
            should_use_candidate_sol=should_use_candidate_sol,
        )
        self._containment_domination_checker = ReachesNewLastPosContainment(
            graph=graph, containment_condition=containment_condition
        )


class ReachesCheaperLastPosSamplingContainment(SamplingContainmentDominationChecker):
    def __init__(
        self,
        graph: Graph,
        num_samples_per_vertex: int,
        should_use_candidate_sol: bool = False,
        containment_condition: int = -1,
    ):
        super().__init__(graph=graph)
        self._sampling_domination_checker = ReachesCheaperLastPosSampling(
            graph=graph,
            num_samples_per_vertex=num_samples_per_vertex,
            should_use_candidate_sol=should_use_candidate_sol,
        )
        self._containment_domination_checker = ReachesCheaperLastPosContainment(
            graph=graph, containment_condition=containment_condition
        )
