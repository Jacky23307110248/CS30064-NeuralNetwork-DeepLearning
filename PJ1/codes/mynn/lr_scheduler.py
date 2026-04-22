from abc import abstractmethod
import numpy as np


class scheduler:
    """Learning-rate scheduler; ``step()`` is invoked once per optimizer step (e.g. each mini-batch)."""

    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0

    @abstractmethod
    def step(self):
        pass


class StepLR(scheduler):
    """
    Multiply ``optimizer.lr`` (current LR) by ``gamma`` every ``step_size`` calls to ``step()``.
    ``optimizer.init_lr`` stays fixed as the reference value from construction.
    Uses ``step_count % step_size == 0`` so decays repeat at ``step_size``, ``2*step_size``, …
    """

    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = max(1, int(step_size))
        self.gamma = float(gamma)

    def step(self) -> None:
        self.step_count += 1
        if self.step_count % self.step_size == 0:
            self.optimizer.lr *= self.gamma


class MultiStepLR(scheduler):
    pass


class ExponentialLR(scheduler):
    """``optimizer.lr *= gamma`` on every ``step()`` call; ``optimizer.init_lr`` is unchanged."""

    def __init__(self, optimizer, gamma=0.95) -> None:
        super().__init__(optimizer)
        self.gamma = float(gamma)

    def step(self) -> None:
        self.optimizer.lr *= self.gamma