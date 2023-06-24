from abc import ABC, abstractmethod


class ConvexSet(ABC):
    """
    Abstract base class for wrappers of convex sets. Implementations will each
    wrap be drake implementations,
    but this allows for other data and methods to the convex set as well.
    """

    @abstractmethod
    def __init__(self):
        pass

    def plot(self, **kwargs):
        """
        Plots the convex set using matplotlib.
        """
        if self.dimension != 2:
            raise NotImplementedError
        options = {"facecolor": "mintcream", "edgecolor": "k", "zorder": 1}
        options.update(kwargs)
        self._plot(**options)

    @property
    @abstractmethod
    def dimension(self):
        pass

    @property
    @abstractmethod
    def set(self):
        pass

    @property
    @abstractmethod
    def center(self):
        pass
