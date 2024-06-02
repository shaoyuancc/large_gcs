import logging

from large_gcs.domination_checkers.ah_containment_domination_checker import (
    AHContainmentDominationChecker,
)

logger = logging.getLogger(__name__)


class ReachesNewContainment(AHContainmentDominationChecker):
    @property
    def include_cost_epigraph(self):
        return False
