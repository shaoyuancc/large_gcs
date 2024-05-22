import logging
from typing import List

from large_gcs.algorithms.search_algorithm import SearchNode
from large_gcs.domination_checkers.ah_containment_domination_checker import (
    AHContainmentDominationChecker,
)

logger = logging.getLogger(__name__)


class ReachesNewContainment(AHContainmentDominationChecker):
    @property
    def include_cost_epigraph(self):
        return False
