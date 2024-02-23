from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import RandomGenerator


class ConvexSet(ABC):
    """
    Abstract base class for wrappers of convex sets. Implementations will each
    wrap be drake implementations,
    but this allows for other data and methods to the convex set as well.
    """

    @abstractmethod
    def __init__(self):
        pass

    def plot(self, mark_center: bool = False, **kwargs):
        """
        Plots the convex set using matplotlib.
        """
        if self.dim != 2:
            raise NotImplementedError
        options = {"facecolor": "mintcream", "edgecolor": "k", "zorder": 1}
        options.update(kwargs)
        plt.gca().set_aspect("equal")
        self._plot(**options)
        if mark_center:
            plt.scatter(*self.center, color="k", zorder=2)

    def get_samples(self, n_samples=100) -> np.ndarray:
        samples = []
        generator = RandomGenerator()
        # Setting the initial guess made sampling in the contact set fail
        # initial_guess = self.set.MaybeGetFeasiblePoint()
        try:
            # samples.append(self.set.UniformSample(generator, initial_guess))
            samples.append(self.set.UniformSample(generator))
            for i in range(n_samples - 1):
                samples.append(
                    self.set.UniformSample(
                        generator, previous_sample=samples[-1], mixing_steps=500
                    )
                )
        except:
            print("Warning: failed to sample convex set")
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
