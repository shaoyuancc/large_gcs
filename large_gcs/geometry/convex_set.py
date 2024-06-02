import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import RandomGenerator

logger = logging.getLogger(__name__)


class ConvexSet(ABC):
    """Abstract base class for wrappers of convex sets.

    Implementations will each wrap be drake implementations, but this
    allows for other data and methods to the convex set as well.
    """

    @abstractmethod
    def __init__(self):
        pass

    def _plot(self, ax=None) -> None:
        raise NotImplementedError("_plot not implemented for" + self.__class__.__name__)

    def plot(self, mark_center: bool = False, ax=None, **kwargs):
        """Plots the convex set using matplotlib."""
        if self.dim > 2:
            raise NotImplementedError
        options = {"facecolor": "mintcream", "edgecolor": "k", "zorder": 1}
        options.update(kwargs)

        if ax is None:
            ax = plt.gca()

        ax.set_aspect("equal")
        self._plot(ax=ax, **options)
        if mark_center:
            ax.scatter(*self.center, color="k", zorder=2)

    def get_samples(self, n_samples=100) -> np.ndarray:
        samples = []
        generator = RandomGenerator()
        # Setting the initial guess made sampling in the contact set fail
        # initial_guess = self.set.MaybeGetFeasiblePoint()
        # logger.debug(f"Initial guess for sampling: {initial_guess}")
        try:
            # samples.append(self.set.UniformSample(generator, initial_guess))
            samples.append(self.set.UniformSample(generator))
            logger.debug(f"Sampled 1 points from convex set")
            for i in range(n_samples - 1):
                samples.append(
                    self.set.UniformSample(
                        # 500
                        generator,
                        previous_sample=samples[-1],
                        mixing_steps=100,
                    )
                )
                logger.debug(f"Sampled {i+2} points from convex set")
        except (RuntimeError, ValueError) as e:
            chebyshev_center = self.set.ChebyshevCenter()
            logger.warning("Failed to sample convex set" f"\n{e}")
            return np.array([chebyshev_center])
        return np.array(samples)

    @property
    @abstractmethod
    def dim(self):
        pass

    @property
    @abstractmethod
    def set(self):
        pass

    @property
    @abstractmethod
    def center(self):
        pass
