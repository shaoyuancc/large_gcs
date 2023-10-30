from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from large_gcs.graph.contact_graph import ContactGraph


@dataclass
class AbstractionModel:
    graphs: List[ContactGraph]
    abs_fns: List[Callable]


class AbstractionModelGenerator:
    @abstractmethod
    def generate(self, concrete_graph: ContactGraph) -> AbstractionModel:
        pass
