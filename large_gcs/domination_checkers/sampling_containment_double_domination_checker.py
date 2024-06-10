import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np
import pypolycontain as pp
from pydrake.all import GurobiSolver, MathematicalProgram

from large_gcs.algorithms.search_algorithm import AlgMetrics, SearchNode
from large_gcs.domination_checkers.ah_containment_last_pos import (
    AHContainmentLastPos,
    ReachesCheaperLastPosContainment,
    ReachesNewLastPosContainment,
)
from large_gcs.domination_checkers.reaches_cheaper_containment import (
    ReachesCheaperContainment,
)
from large_gcs.domination_checkers.reaches_cheaper_sampling import (
    ReachesCheaperSampling,
)
from large_gcs.domination_checkers.reaches_new_containment import ReachesNewContainment
from large_gcs.domination_checkers.reaches_new_sampling import ReachesNewSampling
from large_gcs.domination_checkers.sampling_containment_domination_checker import (
    SamplingContainmentDominationChecker,
)
from large_gcs.domination_checkers.sampling_last_pos import (
    ReachesCheaperLastPosSampling,
    ReachesNewLastPosSampling,
    SamplingLastPos,
)

logger = logging.getLogger(__name__)


class SamplingContainmentDoubleDominationChecker(SamplingContainmentDominationChecker):
    def is_dominated(
        self, candidate_node: SearchNode, alternate_nodes: List[SearchNode]
    ) -> bool:
        sample_is_dominated = self.sample_is_dominated(candidate_node, alternate_nodes)

        if np.all(~sample_is_dominated):
            return False

        AH_n_ns, AH_n_fs = self._maybe_create_both_path_AH_polytopes(candidate_node)
        logger.debug(
            f"Checking domination of candidate node terminating at vertex {candidate_node.vertex_name}"
            f"\n via path: {candidate_node.vertex_path}"
        )
        are_keeping = False
        alt_paths_to_prune = []
        for alt_i, alt_n in enumerate(alternate_nodes):
            AH_alt_ns, AH_alt_fs = self._maybe_create_both_path_AH_polytopes(alt_n)
            if sample_is_dominated[alt_i] and not are_keeping:
                # Check whether candidate is contained in alternate
                logger.debug(
                    f"Checking if candidate node is dominated by alternate node with path:"
                    f"{alt_n.vertex_path}"
                )

                if self.is_contained_in_double(AH_n_ns, AH_n_fs, AH_alt_ns, AH_alt_fs):
                    return True
            elif not sample_is_dominated[alt_i]:
                if self.is_contained_in_double(AH_n_ns, AH_n_fs, AH_alt_ns, AH_alt_fs):
                    alt_paths_to_prune.append(alt_n)
                    are_keeping = True
        # Prune alternate paths
        for alt_n in alt_paths_to_prune:
            alternate_nodes.remove(alt_n)
        self._alg_metrics._S_pruned_counts[candidate_node.vertex_name] += len(
            alt_paths_to_prune
        )

        return False

    def _maybe_create_both_path_AH_polytopes(
        self, node: SearchNode
    ) -> Tuple[pp.objects.AH_polytope, pp.objects.AH_polytope]:
        if node.ah_polyhedron_ns is None:
            (
                node.ah_polyhedron_ns,
                _,
            ) = self._create_path_AH_polytope_from_nullspace_sets(node)
        if node.ah_polyhedron_fs is None:
            node.ah_polyhedron_fs, _ = self._create_path_AH_polytope_from_full_sets(
                node
            )
        return node.ah_polyhedron_ns, node.ah_polyhedron_fs

    def is_contained_in_double(self, AH_n_ns, AH_n_fs, AH_alt_ns, AH_alt_fs):
        def target_function(pipe, func, args):
            try:
                result = func(*args)
                pipe.send(result)
            except Exception as e:
                pipe.send(e)
            finally:
                pipe.close()

        parent_conn1, child_conn1 = multiprocessing.Pipe()
        parent_conn2, child_conn2 = multiprocessing.Pipe()

        p1 = multiprocessing.Process(
            target=target_function,
            args=(child_conn1, self.solve_containment, (AH_n_ns, AH_alt_ns)),
        )
        p2 = multiprocessing.Process(
            target=target_function,
            args=(child_conn2, self.solve_containment, (AH_n_fs, AH_alt_fs)),
        )

        p1.start()
        p2.start()

        first_result = None
        try:
            while True:
                if parent_conn1.poll():  # Check if there is data to read
                    result = parent_conn1.recv()
                    if isinstance(result, Exception):
                        logger.error(f"ns generated an exception: {result}")
                        result = None
                    else:
                        logger.debug(f"ns finished first")
                    first_result = result
                    if p2.is_alive():
                        p2.terminate()  # Terminate the other process
                        logger.debug("Terminated process p2 after ns finished")
                    p2.join()
                    break

                if parent_conn2.poll():  # Check if there is data to read
                    result = parent_conn2.recv()
                    if isinstance(result, Exception):
                        logger.error(f"fs generated an exception: {result}")
                        result = None
                    else:
                        logger.debug(f"fs finished first")
                    first_result = result
                    if p1.is_alive():
                        p1.terminate()  # Terminate the other process
                        logger.debug("Terminated process p1 after fs finished")
                    p1.join()
                    break
        finally:
            if p1.is_alive():
                p1.terminate()
                p1.join()
                logger.debug("Terminated process p1 in cleanup")
            if p2.is_alive():
                p2.terminate()
                p2.join()
                logger.debug("Terminated process p2 in cleanup")

        return first_result

    @staticmethod
    def solve_containment(AH_X, AH_Y):
        prog = MathematicalProgram()
        pp.subset(prog, AH_X, AH_Y, -1)
        solver = GurobiSolver()
        result = solver.Solve(prog)
        return result.is_success()


class ReachesNewSamplingContainmentDouble(
    SamplingContainmentDoubleDominationChecker,
    ReachesNewContainment,
    ReachesNewSampling,
):
    pass


class ReachesCheaperSamplingContainmentDouble(
    SamplingContainmentDoubleDominationChecker,
    ReachesCheaperContainment,
    ReachesCheaperSampling,
):
    pass


class SamplingContainmentLastPosDoubleDominationChecker(
    SamplingContainmentDoubleDominationChecker, AHContainmentLastPos, SamplingLastPos
):
    pass


class ReachesNewLastPosSamplingContainmentDouble(
    SamplingContainmentDoubleDominationChecker,
    ReachesNewLastPosContainment,
    ReachesNewLastPosSampling,
):
    pass


class ReachesCheaperLastPosSamplingContainmentDouble(
    SamplingContainmentDoubleDominationChecker,
    ReachesCheaperLastPosContainment,
    ReachesCheaperLastPosSampling,
):
    pass
