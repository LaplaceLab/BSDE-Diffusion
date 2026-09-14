"""Small reference control network for vector-valued inverse problems."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


class TimeConditionedMLP(nn.Module):
    """Estimate the BSDE martingale integrand from time and state.

    This compact default is suitable for vectors and small tensors. Applications
    with structured high-dimensional states can pass any compatible ``nn.Module``
    to the solver instead.
    """

    def __init__(self, event_shape: Sequence[int], hidden_features: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.event_shape = tuple(int(size) for size in event_shape)
        if not self.event_shape or any(size <= 0 for size in self.event_shape):
            raise ValueError("event_shape must contain positive dimensions")
        if hidden_features <= 0 or depth < 1:
            raise ValueError("hidden_features must be positive and depth must be at least one")

        features = math.prod(self.event_shape)
        layers: list[nn.Module] = [nn.Linear(features + 1, hidden_features), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(hidden_features, hidden_features), nn.SiLU()))
        layers.append(nn.Linear(hidden_features, features))
        self.network = nn.Sequential(*layers)

    def forward(self, time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if tuple(state.shape[1:]) != self.event_shape:
            raise ValueError(f"expected event shape {self.event_shape}, received {tuple(state.shape[1:])}")
        time_column = time.to(device=state.device, dtype=state.dtype).reshape(1, 1).expand(state.shape[0], 1)
        flat_state = state.reshape(state.shape[0], -1)
        return self.network(torch.cat((flat_state, time_column), dim=1)).reshape_as(state)

