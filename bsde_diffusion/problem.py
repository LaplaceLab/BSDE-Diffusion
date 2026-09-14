"""Problem definition for terminal-conditioned diffusion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

Tensor = torch.Tensor
StateField = Callable[[Tensor, Tensor], Tensor]
TerminalMap = Callable[[Tensor], Tensor]
Driver = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


def zero_driver(_time: Tensor, _state: Tensor, value: Tensor, _control: Tensor) -> Tensor:
    """Return the zero driver used by terminal-map inverse problems."""
    return torch.zeros_like(value)


@dataclass(frozen=True, slots=True)
class AssociatedBSDE:
    """A diagonal-noise SDE paired with a terminal-value BSDE.

    Callables receive a scalar time tensor and a batched state tensor. ``drift``
    and ``diffusion`` must return tensors broadcastable to the state. ``terminal``
    must preserve the state shape; it is the task-specific extension point that
    encodes measurement or physics feasibility.
    """

    initial_state: Tensor
    time_grid: Tensor
    drift: StateField
    diffusion: StateField
    terminal: TerminalMap
    driver: Driver = zero_driver

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.initial_state) or self.initial_state.ndim < 2:
            raise ValueError("initial_state must be a batched tensor")
        if self.initial_state.shape[0] != 1:
            raise ValueError("initial_state must contain one shared anchor in its batch dimension")
        if not self.initial_state.is_floating_point():
            raise TypeError("initial_state must use a floating-point dtype")
        if not torch.is_tensor(self.time_grid) or self.time_grid.ndim != 1:
            raise ValueError("time_grid must be a one-dimensional tensor")
        if self.time_grid.numel() < 2:
            raise ValueError("time_grid must contain at least two points")
        if not torch.all(self.time_grid[1:] > self.time_grid[:-1]):
            raise ValueError("time_grid must be strictly increasing")

