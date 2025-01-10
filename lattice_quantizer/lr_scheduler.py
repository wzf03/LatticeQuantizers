from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numba import float64, int32, int64, njit


class LRScheduler(ABC):
    @abstractmethod
    def get_lr_scheduler(self, total_steps: int) -> Callable[[int], float]:
        pass


@njit(float64(int32, int32, float64, float64, float64), cache=True)
def factorized_lr_scheduler(
    step: int, warmup_steps: int, peak_lr: float, end_lr: float, factor: float
) -> float:
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    return max(end_lr, peak_lr * factor ** (step - warmup_steps))


@dataclass
class FactorizedLR(LRScheduler):
    warmup_steps: int
    peak_lr: float
    end_lr: float
    factor: float

    def get_lr_scheduler(self, _total_steps: int) -> Callable[[int], float]:
        warmup_steps = self.warmup_steps
        peak_lr = self.peak_lr
        end_lr = self.end_lr
        factor = self.factor

        @njit
        def scheduler(step: int) -> float:
            return factorized_lr_scheduler(step, warmup_steps, peak_lr, end_lr, factor)

        return scheduler


@njit(float64(int32, float64, float64, int64), cache=True)
def ratio_lr_scheduler(
    step: int, peak_lr: float, ratio: float, total_steps: int
) -> float:
    return peak_lr * ratio ** (-step / (total_steps - 1))


@dataclass
class RatioLR(LRScheduler):
    peak_lr: float
    ratio: float

    def get_lr_scheduler(self, total_steps: int) -> Callable[[int], float]:
        peak_lr = self.peak_lr
        ratio = self.ratio

        @njit
        def scheduler(step: int) -> float:
            return ratio_lr_scheduler(step, peak_lr, ratio, total_steps)

        return scheduler


@njit(float64(int32, int32, float64, float64, float64), cache=True)
def cosine_lr_scheduler(
    step: int, max_update: int, base_lr: float, final_lr: float, warmup_steps: int
) -> float:
    if step < warmup_steps:
        return base_lr * step / warmup_steps

    if step <= max_update:
        progress = (step - warmup_steps) / (max_update - warmup_steps)
        return final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(progress * np.pi))

    return final_lr


@dataclass
class CosineLR(LRScheduler):
    max_update: int
    base_lr: float
    final_lr: float
    warmup_steps: int

    def get_lr_scheduler(self, _total_steps: int) -> Callable[[int], float]:
        max_update = self.max_update
        base_lr = self.base_lr
        final_lr = self.final_lr
        warmup_steps = self.warmup_steps

        @njit
        def scheduler(step: int) -> float:
            return cosine_lr_scheduler(
                step, max_update, base_lr, final_lr, warmup_steps
            )

        return scheduler
